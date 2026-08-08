# ACP (Agent Client Protocol)

`meka acp` speaks the [Agent Client Protocol](https://agentclientprotocol.com/) over stdio so editor / web / messenger clients can drive a meka turn end to end. Where [Interactive Mode](./interactive-mode.md) and [One-Shot Mode](./one-shot-mode.md) are for humans, ACP is for *programs* that want to host meka inside a richer UI: streamed diffs, native apply-buttons, hosted terminals, and slash-command palettes.

This page describes what meka's ACP surface looks like to a client. Editor-specific setup belongs in each editor's own documentation; the protocol contract is the same everywhere.

## Starting an ACP server

```bash
meka acp
```

The process speaks JSON-RPC 2.0 with newline framing on stdio. There is no human-facing prompt; the binary is meant to be spawned by a client that owns the conversation. The client sends `initialize`, then `session/new` (or `session/load` / `session/resume`), then a series of `session/prompt` calls.

A few flags are worth knowing:

| Flag | Effect |
|------|--------|
| `-v` | Logs to stderr at `info` (incoming client identity, session lifecycle). |
| `-vv` | `debug` (per-request JSON-RPC diagnostics). |
| `RUST_LOG=meka=trace` | Trace level. |

On startup, after the client's `initialize` arrives, meka logs `ACP client connected: <name> <version>` so you can confirm the client identity in `-v` mode.

## What meka advertises (`agentCapabilities`)

These are returned in `InitializeResponse.agentCapabilities`:

- **`loadSession: true`**: the client may call `session/load` with any persisted session id.
- **`sessionCapabilities.list`**: the client may call `session/list` to browse the persisted session catalogue (cwd-filtered, cursor-paginated; sub-agent audit sessions are hidden).
- **`sessionCapabilities.resume`**: the client may adopt a persisted session id without replaying history.
- **`sessionCapabilities.fork`**: the client may branch a copy off a persisted session (see [Forking](#forking)). **Unstable** in the protocol.
- **`sessionCapabilities.close`**: the client may release the active session slot.
- **`sessionCapabilities.additionalDirectories`**: the client may send extra workspace roots on `session/new`, `session/load`, and `session/resume` (see [Multi-root workspaces](#multi-root-workspaces)).
- **`promptCapabilities.embeddedContext: true`**: the client may inline @-mentioned file contents as embedded `resource` blocks (see [Prompt turn](#prompt-turn)).
- **`promptCapabilities.image`**: follows the active profile's `vision` flag (default `true`; set `vision = false` in `[providers.<name>]` for a text-only model). When `true`, the client may attach `image` blocks.

`mcpCapabilities` is intentionally **not** advertised. meka is itself an MCP client, but the servers it consumes are configured via meka's own `config.toml` rather than the `mcpServers` field on `session/new`. Advertising HTTP/SSE while silently ignoring the client's array would have been misleading; the marker will return when client-supplied MCP server connections are actually implemented.

`agentInfo` carries meka's name (`"meka"`) and the running binary version.

## What meka consumes (`clientCapabilities`)

The client advertises these in `InitializeRequest.clientCapabilities`; meka stashes them and lets the built-in tools route accordingly:

- **`fs.readTextFile: true`**: `read_file` issues `fs/read_text_file { sessionId, path, line?, limit? }` so the client serves the *in-buffer* view of the file. Image and regex `read_file` modes have no `fs/*` analogue and stay local.
- **`fs.writeTextFile: true`**: `write_file` and `edit_file`'s apply step issue `fs/write_text_file { sessionId, path, content }`. meka still attaches diff metadata to the `tool_call_update` so clients with an apply-diff UI can render it.
- **`terminal: true`**: `execute_command` runs the four-call dance: `terminal/create` → `terminal/wait_for_exit` → `terminal/output` → `terminal/release`. On `session/cancel` or a per-call timeout, meka issues `terminal/kill` and still reads accumulated output. **Exception**: in `read` permission mode meka keeps the local sandboxed shell (Landlock / bwrap / sandbox-exec / Low-Integrity) rather than delegating, so the sandbox isn't bypassed by the client's terminal.
- **`elicitation.form` / `elicitation.url`**: when an MCP server asks the user for input mid-tool-call, meka issues `elicitation/create` so the prompt renders in the editor. The two modes are advertised independently and checked separately — a server asking for a form when only `url` is advertised is declined rather than sent. Without the capability meka declines every elicitation, which is what it did unconditionally before. Elicitations raised inside a sub-agent forward to the parent session, like permission prompts.

If the client omits a capability, the matching tool falls back to local syscalls; the user-visible behaviour is the same as `meka` in the REPL.

### When the client won't serve a path

Editors differ in which paths they will serve: Zed answers only for the project it has open, another client may serve any absolute path. meka models none of these rules. It asks per path and routes on the answer:

- **`ResourceNotFound` (`-32002`)** means the client will not serve this path, so it holds no buffer for it. meka reads or writes the file locally, and a *write* says so in the tool result — the change still appears in that tool call's diff, but not in the editor's buffer or undo history. This is what keeps ACP as capable as the terminal: the agent can read and edit its own skills, prompts, and configuration even though they live outside the project.
- **Any other error** means the client may own the file and hold unsaved changes for it, so the tool call fails instead of routing around the client. Reading on-disk bytes would hand the model a stale view of a file the user is editing, and writing them back would overwrite unsaved work.

The route is chosen once per tool call by the read, not per request: `edit_file` and `write_file` write back through whichever filesystem they read from, so a diff taken from the editor's buffer isn't applied to disk while the buffer keeps the old content. The read is also the more reliable signal — Zed reports an out-of-project path as `ResourceNotFound` on `fs/read_text_file` but as a generic error on `fs/write_text_file`, so a route chosen from the write's own error would never recognise it.

One case can't honour that: a client advertising `fs.readTextFile` but not `fs.writeTextFile` reads for meka and expects meka to do the write, so the edit lands on disk while the client still holds a buffer for the file. The tool result discloses that too, with its own note.

## Session lifecycle

meka holds an in-memory map of `sessionId → SessionEntry`. Any number of sessions can coexist in one `meka acp` process, each with its own cwd, permission level, conversation, cancellation token, and per-session runtime mutex. Prompts on different sessions run in parallel; a second `session/prompt` for a session that already has one in flight is rejected with `InvalidParams`. The session row is also locked on disk (the same lock the REPL uses), so two `meka` processes can't simultaneously write events for the same session id.

- **`session/new { cwd, mcpServers }`**: mints a fresh persisted session, captures the cwd, takes the on-disk session lock, returns the session id and the current `SessionMode` state.
- **`session/load { sessionId, cwd, mcpServers }`**: replays the persisted conversation as a stream of `session/update` notifications (`user_message_chunk`, `agent_message_chunk`, `agent_thought_chunk`, `tool_call`, `tool_call_update`) before the response. Orphan tool calls (the persisted log stopped mid-tool) are closed out with a `failed` status so the client's UI doesn't render a stuck spinner. If the client's `cwd` differs from the persisted one, meka updates the persisted row to match; the client wins.
- **`session/list { cwd?, cursor? }`**: paginated index. Filtered to the requested cwd when present; sub-agent sessions are always hidden. `nextCursor` is opaque; round-trip it back to keep paging.
- **`session/resume { sessionId, cwd, mcpServers }`**: adopts the session id without replaying. Use this when the client already has the history rendered. Same cwd-update behaviour as `session/load`.
- **`session/fork { sessionId, cwd, additionalDirectories, mcpServers }`**: copies the session's conversation into a new persisted session, adopts the copy as active, and returns its id. The source is left open and untouched. See [Forking](#forking).
- **`session/close { sessionId }`**: cancels any in-flight prompt, releases the on-disk session lock, and removes the entry from the map.
- **`session/cancel { sessionId }`**: interrupts the active `session/prompt`. The response carries `stopReason: "cancelled"`. If a cancel arrives between turns (after one prompt completed and before the next is installed), meka latches the signal and cancels the next prompt immediately on arrival.
- **`session/set_mode { sessionId, modeId }`**: flips the agent's `Permission` cell. Modes outside `[permissions].enabled` from the config become JSON-RPC errors. On success, meka emits `session/update: current_mode_update`. The flip is atomic and applies to the *next* tool call within an in-flight turn; no need to wait for the turn to finish.

## Prompt turn

A `session/prompt` carries a `prompt` array of `ContentBlock`s. meka accepts:

- **`text`**: the baseline.
- **`resource_link`**: flattened into a `<resource_link name="…" uri="…">description</resource_link>` tag inside the prompt text so the model sees the reference; meka does not fetch the resource server-side.
- **`resource`** (embedded @-mention contents): a text resource is inlined as a `<resource uri="…">…contents…</resource>` tag; a binary (blob) resource becomes a self-closing `<resource uri="…" encoding="base64"/>` marker (the payload is not inlined).
- **`image`**: accepted only when the profile has vision on. The payload is normalized through meka's image pipeline (size cap, format conversion) and forwarded to the model as native vision input (Claude `image`, OpenAI chat `image_url`, Codex `input_image`).

`audio` blocks (and `image` when `vision = false`) produce `InvalidParams`.

Images travel in the other direction too: when a tool looks at one (`read_file` on an image file,
`render_image`, `fetch_url` on an image URL), the picture is forwarded on that tool call as an
`image` content block rather than a placeholder, so the client renders what the model was shown.

While the turn runs, meka streams `session/update` notifications:

- `agent_message_chunk` for each piece of assistant text.
- `agent_thought_chunk` for thinking blocks (Claude OAuth / extended-thinking models).
- `tool_call` when a tool starts, with `kind`, `status: "in_progress"`, an absolute `locations` array (relative paths resolved against the session cwd, with the start line for `read_file`), the raw input, and a human-readable `title`. The title is the tool's primary argument, so editors show what's running rather than the bare tool name: the command for `execute_command`, `Read <path>` / `Edit <path>` / `Write <path>` for file tools, `Fetch <url>`, `Web search: <query>`, etc.
- `tool_call_update` when a tool finishes, with the final `status` (`completed` / `failed`), a `content` array, and `raw_output` (the structured tool result). `execute_command` output is wrapped in a fenced `console` code block so editors render it monospaced; `edit_file` and `write_file` populate diff content blocks so clients can render the apply-diff UI. (Large outputs offloaded to the scratchpad show the scratchpad reference rather than the full payload.)
- `plan` whenever the agent's `todo` tool updates the task list, so clients with a plan panel (e.g. Zed) render the live to-do list. meka's `cancelled` todo status maps to `completed`.
- `session_info_update` once per session, carrying the title (the first user message preview) so a freshly created or loaded tab gets a label without a `session/list` call.
- `usage_update` after each turn, carrying `used` (tokens currently in context: all input tiers plus output) and `size` (the model's context window), so clients with a context gauge (e.g. Zed) show how full the window is. Emitted only once both values are known.
- The `session/prompt` *response* additionally carries `usage`: session-cumulative `totalTokens` / `inputTokens` / `outputTokens` / `cachedReadTokens` / `cachedWriteTokens`. This is the running total for the session, not the gauge — `usage_update` answers "how full is the window", `usage` answers "what has this session cost". `thoughtTokens` is omitted because meka doesn't meter reasoning separately from output.

The response carries a final `stopReason`:

| `stopReason` | Meaning |
|--------------|---------|
| `end_turn` | The agent finished cleanly. |
| `max_tokens` | The provider stopped because the model hit its maximum output tokens. The assistant message may be truncated. |
| `cancelled` | `session/cancel` interrupted the turn, including the case where the cancel caused an exception in an underlying operation. meka probes the per-session cancellation token after `run_turn`; any error returned while the token has fired surfaces as `cancelled` rather than a generic JSON-RPC error. |
| `refusal` | The model declined to comply (Claude `stop_reason: "refusal"` and the OpenAI equivalent). The assistant message contains the refusal text. |

## Permission modes

meka's `Permission` levels map 1:1 to ACP `SessionMode` ids:

| Permission | Mode id | Display name | Description |
|------------|---------|--------------|-------------|
| `None` | `none` | None | No tools available. |
| `Read` | `read` | Read | File reads and searches only. No writes, no shell. |
| `Ask` | `ask` | Ask | Every write or shell command requires approval. |
| `Write` | `write` | Write | All tools allowed without per-call approval. |

The full mode picker is advertised on every session-creation response (`NewSessionResponse.modes`, `LoadSessionResponse.modes`, `ResumeSessionResponse.modes`) but only the modes in `[permissions].enabled` from your `config.toml` are listed; picking a disabled mode would just error.

When the active mode is `ask`, write-gated tools trigger a `session/request_permission` round-trip. Clients render four options:

- **Allow**: run this call only.
- **Always allow**: run this call and skip the prompt for the same tool for the rest of the session.
- **Deny**: refuse this call only.
- **Always deny**: refuse this call and every subsequent call to the same tool.

Sticky decisions live in meka's process memory; they reset on session close.

## Slash commands

Two kinds of slash command are advertised through `session/update: available_commands_update` (after `session/new` / `session/load` / `session/resume`, and refreshed at the top of every `session/prompt` so a skill installed mid-session shows up without a reconnect):

- **Built-in local commands** — `/status` (model, effort, context usage, tokens, mode) and `/mcp` (configured MCP servers and their connection status). They render text back as an `agent_message_chunk` and end the turn immediately, with no model call.
- **Skills** (see [Skills](./skills.md)) — each installed skill is a top-level command carrying a free-form input hint (`"additional context (optional)"`).

When the user picks one from the palette, the client typically inserts `/<name> ` and lets the user type extra context. meka parses the prompt as follows:

- A built-in local command (`/status`, `/mcp`): handled agent-side, output streamed back, turn ends with no model call. Checked first, so a skill can't shadow a built-in (a skill named `status`/`mcp` is dropped from the palette).
- Plain text (no leading slash): passes through to the model unchanged.
- `/<skill-name>` matching an installed skill: loads the skill body via the same path as the REPL's `/skill` command (substituting `${MEKA_SKILL_DIR}` and `${MEKA_SESSION_ID}`) and prepends any extra context the user typed.
- Slash with a syntactically valid but unknown skill name (`/nonexistent`): JSON-RPC error.
- Slash with content that isn't a valid skill identifier (`/etc/hosts`, `//comment`): passes through to the model unchanged, so pasted paths and code comments don't get intercepted.

## Sub-agents

`spawn_agent` and skill-based delegation produce a sub-agent that runs through `PermissionForwardingFrontend`. The sub-agent's own output isn't streamed to the client (its final report flows back through the parent's `tool_call_update`), but its permission prompts, fs delegates, and terminal delegates all forward through the parent's connection, so the editor's apply-diff UI sees a sub-agent's writes the same as the main agent's.

ACP has no sub-agent primitive — no nested sessions, no nested tool calls — so a sub-agent is one tool call, and its progress is that call's content. While it runs, each tool call it starts is appended to a rolling list (the last 20) and pushed as a `tool_call_update` on the parent's `spawn_agent` call, so a long delegated task shows what it is currently doing instead of an opaque spinner. The whole list is resent on each update because clients replace a tool call's content rather than appending to it. A nested sub-agent's list is not forwarded further up: it already appears as a `spawn_agent` line in its parent's list, and two writers on one tool call's content would overwrite each other.

## Multi-root workspaces

An editor whose workspace holds several folders (Zed's Add Folder to Project) sends the first as `cwd` and the rest as `additionalDirectories`. Clients only send them when the agent advertises `sessionCapabilities.additionalDirectories`, so before meka advertised it every folder but the first was silently dropped and the agent would report files in them as missing.

What the extra roots do and don't change:

- **Search sweeps all of them.** `find_files` and `search_contents` walk every root when you don't pass an explicit `path`. The 60-second walk budget is shared across the whole call, not granted per root, so a four-folder workspace doesn't get a four-minute ceiling. Passing `path` searches exactly that tree, as before.
- **A truncated `search_contents` says which roots it skipped.** Roots are walked in order starting from `cwd`, so a busy `cwd` can fill the 100-match cap before later roots are reached. When that happens the output names how many roots went unsearched, rather than leaving their absence to read as "nothing there". Pass `path` to search one directly, or `scratchpad` to lift the cap. `find_files` is unaffected: its cap bounds only what it prints, so it still counts matches across every root.
- **Overlapping roots are collapsed.** A root nested inside another (or a repeat of `cwd`) is dropped, so its tree isn't walked twice and its files aren't reported twice. Symlinked duplicates aren't detected.
- **The model is told they exist.** Each root is named in the per-turn environment context, alongside the working directory.
- **Relative paths still resolve against `cwd` only.** This is what the spec requires: `cwd` "remains the base for relative paths". Use an absolute path to reach a file in another root.
- **The shell still runs in `cwd`.** `execute_command` is unaffected.
- **A stale root is skipped, not fatal.** A root that no longer exists is passed over so the other roots can still answer; `search_contents` reports "does not exist" only when *no* root existed. Root paths are escaped before they reach the glob engine, so a folder named `2024*` or `notes[1]` matches literally instead of widening the search.

Every entry must be an absolute path; a relative one is rejected with `InvalidParams`.

The list is persisted and reported back on `session/list` as `SessionInfo.additionalDirectories`, which is how a client rebuilds the workspace shape when you pick a session out of its history. `session/load` and `session/resume` carry the *complete resulting* list, so they replace what was stored rather than merging: reopening a session from a window that no longer has the second folder correctly narrows it, and an empty list clears the roots.

## Forking

`session/fork` branches a copy off a persisted session: the new session starts with the source's full conversation and continues from there, while the source stays open and unchanged. It's the protocol's way to explore a direction, or run something like a summary, without writing into the conversation the user is looking at.

The request is a session-*creation* request, not a row copy: it carries its own `cwd` and `additionalDirectories`, and meka applies those to the fork rather than inheriting the source's. `mcpServers` is ignored, as on `session/new`. The response returns the new `sessionId` and the current `SessionMode` state, and the fork is registered as active immediately, so it can be prompted without a further `session/load` or `session/resume`.

There is no replay: unlike `session/load`, forking emits no `session/update` stream for the copied history, since a client that just forked already has the transcript rendered.

Sub-agent child transcripts are not copied, and the fork records no link back to its source. See [Forking a Session](./sessions.md#forking-a-session) for the full semantics.

This method is marked **unstable** in the protocol: it is not part of the spec yet and may change or be removed. Zed does not currently call it.

## Known limitations

- **Tool-call diff metadata isn't persisted.** A session reopened with `session/load` replays `tool_call_update`s as plain text rather than diffs. The on-disk content is unaffected.
- **`read` mode + `terminal` capability**: meka runs the local sandboxed shell instead of delegating, to preserve the read-only jail. The shell appears in meka's own output rather than the client's terminal pane until you switch to `ask` or `write`.
- **Image and regex `read_file`**: stay local. The `fs/read_text_file` request carries only text, so there's no protocol surface to delegate either case.
- **`audio` prompts**: not supported; `audio` content blocks produce `InvalidParams`.
- **No client-side model gate for images**: when `vision` is on, meka forwards images to whatever model the profile names; a non-vision model returns a provider error rather than meka rejecting up front. Set `vision = false` for text-only endpoints.
