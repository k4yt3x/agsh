# Tools Overview

Tools are the actions that the agent can perform on your behalf. The LLM decides which tools to call based on your instructions.

## Available Tools

| Tool | Permission | Description |
|------|-----------|-------------|
| [`read_file`](./file-operations.md#read_file) | Read | Read file contents |
| [`edit_file`](./file-operations.md#edit_file) | Write | Make string replacements in a file |
| [`write_file`](./file-operations.md#write_file) | Write | Create or overwrite a file |
| [`find_files`](./search.md#find_files) | Read | Find files by glob pattern |
| [`search_contents`](./search.md#search_contents) | Read | Search file contents with regex |
| [`fetch_url`](./web.md#fetch_url) | Read | Fetch a web page as markdown |
| [`web_search`](./web.md#web_search) | Read | Search the web |
| [`execute_command`](./shell.md#execute_command) | Read/Write | Run a shell command |
| [`todo`](./overview.md#todo) | Read | Manage and read a structured task list |
| [`spawn_agent`](./overview.md#spawn_agent) | Read | Delegate tasks to a sub-agent |
| [`scratchpad_write`](./scratchpad.md#scratchpad_write) | Read | Store content in the scratchpad |
| [`scratchpad_read`](./scratchpad.md#scratchpad_read) | Read | Read a scratchpad entry |
| [`scratchpad_edit`](./scratchpad.md#scratchpad_edit) | Read | Edit a scratchpad entry |
| [`scratchpad_list`](./scratchpad.md#scratchpad_list) | Read | List scratchpad entries |
| [`scratchpad_delete`](./scratchpad.md#scratchpad_delete) | Read | Delete a scratchpad entry |
| [`skill`](./overview.md#skill) | Read | Load a named skill's instructions |
| [`memory_write`](../usage/memory.md) | Read | Save a durable note that outlives the session |
| [`memory_read`](../usage/memory.md) | Read | Load one saved memory in full |
| [`memory_search`](../usage/memory.md) | Read | Regex over the full text of every memory |
| [`memory_delete`](../usage/memory.md) | Read | Delete a saved memory |
| [`render_image`](./overview.md#render_image) | Read | View an image from in-memory base64 or scratchpad |
| [`recall`](./overview.md#recall) | Read | Search the full conversation history, including compacted turns |
| [`recall_read`](./overview.md#recall) | Read | Read conversation turns by index |
| [`schedule_create`](../usage/scheduling.md) | Read | Schedule a future turn for this session |
| [`schedule_list`](../usage/scheduling.md) | Read | List this session's scheduled jobs |
| [`schedule_cancel`](../usage/scheduling.md) | Read | Cancel a scheduled job |
| [`task_list`](../usage/background.md) | Read | List this session's background tasks |
| [`task_cancel`](../usage/background.md) | Read | Stop a running background task |

The `schedule_*` tools require [`[schedule] enabled`](../configuration/config-file.md#schedule) (on by default) and the `task_*` tools require [`[background] enabled`](../configuration/config-file.md#background) (off by default). A disabled subsystem registers no tools at all, rather than shipping schemas that could only fail.

## Permission Requirements

Tools are grouped by the minimum permission level required:

**Read permission** (available in read, ask, and write modes):
- `read_file`, `find_files`, `search_contents`, `fetch_url`, `web_search`
- `execute_command` (sandboxed, filesystem write-protected)
- `todo`, `spawn_agent`, `skill`, `render_image`
- `recall`, `recall_read`
- All scratchpad tools
- All memory tools. Writing a memory needs only read permission: the store is meka's own, under
  its config directory, not your working tree

**Write permission** (only available in write mode):
- `edit_file`, `write_file`, `execute_command` (unsandboxed)

In **ask** mode, all tools are available but each call requires user confirmation.

In **none** mode, no tools are available. The agent can only respond with text.

## Filtering Built-in Tools

Any built-in can be allow-listed, blocked, or have its required permission overridden via the `[tools]` table in `config.toml`. See [`[tools]`: built-in tool filters](../configuration/config-file.md#tools-built-in-tool-filters). Run `meka tools list` to see every built-in with its effective permission and current status.

## MCP Tools

When [MCP servers](../configuration/config-file.md#mcp-servers-mcp) are configured, their tools are registered under a namespaced name of the form `mcp__<server>__<tool>` (e.g. `mcp__notion__notion-search`). The `mcp__` prefix matches [Claude Code](https://github.com/anthropics/claude-code)'s convention and keeps MCP tools from colliding with built-in names. They appear in the per-turn context catalogue alongside the built-ins, with their resolved permission level annotated inline, and are called the same way.

meka also exposes seven built-in **MCP meta-tools** for browsing server-side resources and prompts. All are deferred by default; call `load_tool` with the exact name to make the schema available on the next turn:

| Tool | Permission | Description |
|------|-----------|-------------|
| `list_mcp_resources` | Read | List resources a server exposes |
| `read_mcp_resource` | Read | Read a server resource by URI |
| `list_mcp_prompts` | Read | List server-defined prompts |
| `get_mcp_prompt` | Read | Render a server prompt with arguments |
| `subscribe_mcp_resource` | Read | Receive change notifications for a resource |
| `unsubscribe_mcp_resource` | Read | Stop receiving change notifications |
| `list_mcp_resource_updates` | Read | Inspect pending resource-change notifications |

## Deferred Tools

Most MCP tools are **deferred**: they are registered and listed under `[Tool discovery]` in the per-turn context, but their JSON schemas are withheld from the request until the agent calls `load_tool`. A large server can advertise fifty tools with multi-kilobyte schemas, and shipping all of them on every turn costs more than it returns.

The trade-off is that until a tool is loaded, the agent sees only its name and a summary clipped to 250 characters. **Anything past that clip is invisible**, including optional parameters, and a summary that was clipped ends in `…`.

Two behaviours exist so this never turns into a silent wrong answer:

- Calling a deferred tool without loading it **works**. The agent may be confident about the required arguments, and forcing a round trip it doesn't need is worse than allowing it.
- But when it does that and the tool has documented parameters it didn't pass, meka appends a note to the tool result naming them, with their types, defaults, and descriptions. A wrong default stops being invisible. The note is emitted once per tool per run.

`load_tool` takes one name or an array of up to ten, so a task needing several tools off one server costs one round trip:

```text
load_tool({"name": ["mcp__notion__search", "mcp__notion__fetch"]})
```

Tools listed in a server's [`eager_load_tools`](../configuration/config-file.md#mcp-servers) skip all of this: their schemas ship from turn 1. Use it for tools whose optional parameters matter and that the agent reaches for constantly.

**When writing a tool description for a server meka will consume**, put whatever a caller must know to use the tool correctly in the first two sentences. That may be all anyone ever sees.

## Background Calls

With [`[background] enabled`](../configuration/config-file.md#background), every tool gains an optional `background` parameter, MCP tools included. A call that sets it returns a task id immediately and delivers its result later as its own turn, which is what makes a twenty-minute build affordable. See [Background Tasks](../usage/background.md).

```text
execute_command({"command": "cargo test --all", "background": true})
```

Like `scratchpad`, `background` is meka's own: it is consumed by the agent loop and removed from the arguments before the tool, or a remote MCP server, ever sees it.

A tool that advertises `background` itself keeps it. meka does not splice its own parameter over a name a tool already uses, and does not strip or interpret one either, so a server with a `background` colour or a detach flag of its own receives the argument untouched and the call does not detach.

These two are also the only parameters meka type-checks. A `background` that is not a boolean, or a `scratchpad` that is not a string, refuses the call and says what was expected, rather than being read as absent. Both decide what a call *does* rather than what it is called with, so ignoring a wrong type would silently turn a detached call into a blocking one, or drop output the agent asked to keep. A tool's own arguments are the tool's to validate: meka reports a mismatch as an advisory on the result and lets the call through, since a remote server is the authority on what it accepts. `null` counts as absent for both, which is what models emit for an optional argument they are not using.

## Scratchpad Parameter

All tools support an optional `scratchpad` string parameter. When provided, the tool's output is saved to the scratchpad under that name instead of being returned inline. This lets the agent store large outputs for later processing without consuming conversation context.

```text
execute_command({"command": "pdftotext doc.pdf -", "scratchpad": "pdf_text"})
```

## How Tool Calls Work

1. The agent receives your instruction and decides which tools to call
2. For each tool call, meka checks the current permission level
3. In ask mode, you are prompted to approve or deny each tool call
4. If permitted, the tool executes and its output is fed back to the agent
5. The agent may make additional tool calls or respond with text
6. This loop continues until the agent has no more tool calls to make

Tool calls and their results are displayed in the terminal so you can see what the agent is doing.

## `todo`

A built-in tool for managing a structured task list during a session. The agent uses it to track multi-step work and communicate progress; the list is displayed in the terminal (for the main agent) and injected into the conversation context each turn. Every call returns the full current list (with task numbers), so the agent never needs a separate read.

Inputs (all optional):

- `title` — a short heading summarizing the overall goal; rendered as the list's heading (`TODO: <title>`). **Required whenever you pass `items`**, and persists across later `set` updates.
- `items` — replace the whole list. Each entry is a task string (status defaults to `pending`) or an object `{text, status}`. Tasks are numbered `1..N` in order.
- `set` — a sparse status update keyed by task number, e.g. `{"1": "completed", "2": "in_progress"}`. This is the common path while working.

Task statuses are `pending`, `in_progress`, `completed`, and `cancelled`. Calling `todo` with no arguments simply reads the current list.

## `spawn_agent`

Spawns a sub-agent to perform research, analysis, or any other delegated task. The sub-agent gets its own private todo list (`todo` operates on the sub-agent's own state), runs silently (its tool calls are not surfaced to the terminal), and returns a single text report. Use this to keep exploratory or speculative work out of the main conversation context.

Multiple `spawn_agent` calls in one assistant turn run in parallel; useful when independent investigations can proceed concurrently.

**Recursion.** Sub-agents may themselves spawn further sub-agents, so an agent can orchestrate a team. Nesting is bounded by [`session.subagent_max_depth`](../configuration/config-file.md#sessionsubagent_max_depth) (default 3; `1` reproduces the old "sub-agents can't spawn" behavior, `0` disables `spawn_agent` entirely). Pass the optional `max_depth` parameter to tune how deep a given subtree may recurse; a built-in absolute cap always bounds real nesting so recursion can't run away.

**Permission.** By default a sub-agent inherits the parent's permission level. Pass the optional `permission` parameter (`none` / `read` / `ask` / `write`) to run it at a *more restricted* level: the value is clamped to the parent's level as a ceiling, so a sub-agent can never be escalated above its parent. This lets a write-mode orchestrator hand untrusted or risky work to a read-only sub-agent.

## `skill`

Loads a named skill's instructions. Skills are user-defined knowledge packages stored in `~/.config/meka/skills/<name>/SKILL.md`. The per-turn context lists available skills with their description and when-to-use hint; the agent calls `skill({"name": "<skill-name>"})` to load the full body. See [Skills](../usage/skills.md) for how to author skills.

## `render_image`

Displays an image the agent has in memory, as base64 bytes or in a scratchpad entry, as a multimodal content block. Complements `fetch_url` (network) and `read_file` (local file) by covering the third case: image data produced on the fly by a command pipeline.

Typical workflow:

```text
execute_command({"command": "ffmpeg -i input.mp4 -vframes 1 -f image2pipe pipe: | base64 -w0", "scratchpad": "frame"})
render_image({"from_scratchpad": "frame"})
```

Parameters:

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `from_scratchpad` | string | one of two | Name of a scratchpad entry containing base64-encoded image bytes |
| `base64` | string | one of two | Base64-encoded image bytes, passed inline |

Exactly one of `from_scratchpad` or `base64` must be provided. Prefer `from_scratchpad` for large images; inline base64 inflates tool-call JSON.

The bytes must decode to a supported raster image. PNG, JPEG, GIF, WebP, and BMP pass through unchanged; TIFF, ICO, HDR, EXR, TGA, PNM, QOI, DDS, and Farbfeld are auto-converted to PNG. Size cap is ~3.75 MB on the final payload.

Only call `render_image` when the current model supports vision input.

## `recall` / `recall_read`

Search and re-read this session's **full** conversation, including earlier turns that [compaction](../usage/interactive-mode.md#compact) summarized and removed from the model's context. Compaction never deletes turns (it appends a boundary and hides the older ones); these tools read straight from the on-disk event log, so a detail the compaction summary dropped is still recoverable.

`recall` searches and returns matching lines, each tagged with a message index (`#N`) and role:

```text
recall({"query": "auth token", "regex": false, "limit": 20})
```

- `query` (required) — text to search for; a literal substring (case-insensitive) unless `regex` is set.
- `regex` — treat `query` as a case-sensitive regular expression. Default: `false`.
- `limit` — maximum matches to return (capped at 100). Default: 20.

`recall_read` reads turns by the `#N` index that `recall` reports:

```text
recall_read({"start": 47, "count": 3})
```

- `start` (required) — 1-based message index to read from.
- `count` — number of consecutive messages to read (max 20). Default: 1.
- `scratchpad` — save the output to a scratchpad entry instead of returning it inline.

After a compaction, the summary message reminds the agent that these tools exist. Large tool outputs appear as `<large-output>` references in both `recall` and `recall_read` results (rather than inlining the full payload); read their full content with `scratchpad_read`.

## Redirecting output to the scratchpad

Several tools (`execute_command`, `find_files`, `search_contents`, `fetch_url`, `spawn_agent`) accept an optional `scratchpad` parameter that redirects their output to a named scratchpad entry instead of returning it inline. When this parameter is set, the tool produces its **full, untruncated output**: internal result-count caps (`find_files` 200, `search_contents` 100) and length caps (`fetch_url` `max_length`) are lifted for the scratchpad-bound result.
