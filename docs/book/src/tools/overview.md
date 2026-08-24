# Tools Overview

Tools are the actions that the agent can perform on your behalf. The LLM decides which tools to call based on your instructions.

## Available Tools

| Tool | Permission | Description |
|------|-----------|-------------|
| [`read_file`](./file-operations.md#read_file) | Read | Read file contents |
| [`edit_file`](./file-operations.md#edit_file) | Workspace | Make string replacements in a file |
| [`write_file`](./file-operations.md#write_file) | Workspace | Create or overwrite a file |
| [`find_files`](./search.md#find_files) | Read | Find files by glob pattern |
| [`search_contents`](./search.md#search_contents) | Read | Search file contents with regex |
| [`fetch_url`](./web.md#fetch_url) | Read | Fetch a web page as markdown |
| [`search_web`](./web.md#search_web) | Read | Search the web |
| [`execute_command`](./shell.md#execute_command) | Read | Run a shell command (see the note below) |
| [`todo`](./overview.md#todo) | Read | Manage and read a structured task list |
| [`agent_spawn`](./overview.md#agent_spawn) | Read | Delegate tasks to a sub-agent |
| [`agent_list`](./overview.md#agent_list--agent_followup--agent_delete) | Read | List the sub-agents this session spawned |
| [`agent_followup`](./overview.md#agent_list--agent_followup--agent_delete) | Read | Ask a sub-agent another question |
| [`agent_delete`](./overview.md#agent_list--agent_followup--agent_delete) | Read | Discard a sub-agent and its records |
| [`scratchpad_write`](./scratchpad.md#scratchpad_write) | Read | Store content in the scratchpad |
| [`scratchpad_read`](./scratchpad.md#scratchpad_read) | Read | Read a scratchpad entry |
| [`scratchpad_edit`](./scratchpad.md#scratchpad_edit) | Read | Edit a scratchpad entry |
| [`scratchpad_list`](./scratchpad.md#scratchpad_list) | Read | List scratchpad entries |
| [`scratchpad_delete`](./scratchpad.md#scratchpad_delete) | Read | Delete a scratchpad entry |
| [`scratchpad_load_file`](./scratchpad.md#scratchpad_load_file) | Read | Load a file into the scratchpad |
| [`scratchpad_save_file`](./scratchpad.md#scratchpad_save_file) | Workspace | Write a scratchpad entry out to a path |
| [`skill_read`](./overview.md#the-skill_-tools) | Read | Load a named skill's instructions |
| [`skill_search`](./overview.md#the-skill_-tools) | Read | Regex over the full text of every skill |
| [`skill_write`](./overview.md#the-skill_-tools) | Read | Create or update a skill |
| [`skill_delete`](./overview.md#the-skill_-tools) | Read | Delete a skill and its directory |
| [`memory_write`](../usage/memory.md) | Read | Save a durable note that outlives the session |
| [`memory_read`](../usage/memory.md) | Read | Load one saved memory in full |
| [`memory_search`](../usage/memory.md) | Read | Ranked full-text search over every memory |
| [`memory_delete`](../usage/memory.md) | Read | Delete a saved memory |
| [`render_image`](./overview.md#render_image) | Read | View an image from in-memory base64 or scratchpad |
| [`context_check`](./overview.md#context_check--context_compact) | Read | Measure the context window live: occupancy, headroom, compaction count |
| [`context_compact`](./overview.md#context_check--context_compact) | Read | Ask for a compaction at the end of this turn |
| [`conversation_search`](./overview.md#conversation_search--conversation_read) | Read | Search the full conversation history, including compacted turns |
| [`conversation_read`](./overview.md#conversation_search--conversation_read) | Read | Read conversation turns by index |
| [`schedule_create`](../usage/scheduling.md) | Read | Schedule a future turn for this session |
| [`schedule_list`](../usage/scheduling.md) | Read | List this session's scheduled jobs |
| [`schedule_cancel`](../usage/scheduling.md) | Read | Cancel a scheduled job |
| [`task_list`](../usage/background.md) | Read | List this session's background tasks |
| [`task_cancel`](../usage/background.md) | Read | Stop a running background task |

The `schedule_*` tools require [`[schedule] enabled`](../configuration/config-file.md#schedule) (on by default) and the `task_*` tools require [`[background] enabled`](../configuration/config-file.md#background) (off by default). `skill_write` and `skill_delete` require [`[skills] agent_managed`](../configuration/config-file.md#skills) (off by default) and are never given to a sub-agent. A disabled subsystem registers no tools at all, rather than shipping schemas that could only fail.

## Permission Requirements

Tools are grouped by the minimum permission level required:

**Read permission** (available at `read` and above):
- `read_file`, `find_files`, `search_contents`, `fetch_url`, `search_web`
- `execute_command` (sandboxed, filesystem write-protected)
- `todo`, `agent_spawn`, `agent_list`, `agent_followup`, `agent_delete`, `render_image`
- All skill tools, including `skill_write` and `skill_delete` when they are enabled: like memory, the
  store is meka's own under its config directory, not your working tree
- `conversation_search`, `conversation_read`, `context_check`, `context_compact`
- Every scratchpad tool except `scratchpad_save_file`, which writes to a path you name and so sits at `workspace` with `write_file`
- All memory tools. Writing a memory needs only read permission: the store is meka's own, in
  meka's database, not your working tree.

**Workspace permission** (available at `workspace` and above; writes are confined to the workspace roots at `workspace`):
- `edit_file`, `write_file`, `scratchpad_save_file`

`execute_command` is not in that list: it asks for `read` when a sandbox backend is available and `unrestricted` when none is, so it is reachable at `read` and confined by the *level*, not by its own requirement.

In **ask** mode, all tools are available but each call requires user confirmation. Once approved, nothing is confined: `execute_command` runs unsandboxed and a write reaches anywhere, the same as an approved `write_file`. The prompt is the whole gate. Use `workspace` when you want a boundary instead of a prompt.

In **none** mode, no tools are available. The agent can only respond with text.

## Filtering Built-in Tools

Any built-in can be allow-listed, blocked, or have its required permission overridden via the `[tools]` table in `config.toml`. See [`[tools]`: built-in tool filters](../configuration/config-file.md#tools-built-in-tool-filters). Run `meka tools list` to see every built-in with its effective permission and current status.

## MCP Tools

When [MCP servers](../configuration/config-file.md#mcpservers) are configured, their tools are registered under a namespaced name of the form `mcp__<server>__<tool>` (e.g. `mcp__notion__notion-search`). The `mcp__` prefix matches [Claude Code](https://github.com/anthropics/claude-code)'s convention and keeps MCP tools from colliding with built-in names. They appear in the per-turn context catalogue alongside the built-ins, with their resolved permission level annotated inline, and are called the same way.

meka also exposes seven built-in **MCP meta-tools** for browsing server-side resources and prompts. All are deferred by default; call `load_tool` with the exact name to make the schema available on the next turn:

| Tool | Permission | Description |
|------|-----------|-------------|
| `mcp_resource_list` | Read | List resources a server exposes |
| `mcp_resource_read` | Read | Read a server resource by URI |
| `mcp_prompt_list` | Read | List server-defined prompts |
| `mcp_prompt_get` | Read | Render a server prompt with arguments |
| `mcp_resource_subscribe` | Read | Receive change notifications for a resource |
| `mcp_resource_unsubscribe` | Read | Stop receiving change notifications |
| `mcp_resource_updates_list` | Read | Inspect pending resource-change notifications |

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

Tools listed in a server's [`eager_load_tools`](../configuration/config-file.md#mcpservers) skip all of this: their schemas ship from turn 1. Use it for tools whose optional parameters matter and that the agent reaches for constantly.

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

## `agent_spawn`

Spawns a sub-agent to perform research, analysis, or any other delegated task. The sub-agent gets its own private todo list (`todo` operates on the sub-agent's own state), runs silently (its tool calls are not surfaced to the terminal), and returns a single text report. Use this to keep exploratory or speculative work out of the main conversation context.

Multiple `agent_spawn` calls in one assistant turn run in parallel; useful when independent investigations can proceed concurrently.

**Recursion.** Sub-agents may themselves spawn further sub-agents, so an agent can orchestrate a team. Nesting is bounded by [`session.subagent_max_depth`](../configuration/config-file.md#sessionsubagent_max_depth) (default 3; `1` reproduces the old "sub-agents can't spawn" behavior, `0` disables `agent_spawn` entirely). Pass the optional `max_depth` parameter to tune how deep a given subtree may recurse; a built-in absolute cap always bounds real nesting so recursion can't run away.

**Permission.** By default a sub-agent inherits the parent's permission level. Pass the optional `permission` parameter (`none` / `read` / `workspace` / `ask` / `unrestricted`) to run it at a *more restricted* level: the value is clamped to the parent's level as a ceiling, so a sub-agent can never be escalated above its parent. This lets an orchestrator hand untrusted or risky work to a read-only sub-agent. `workspace` and `ask` are incomparable, so asking for one under a parent holding the other yields the parent's own level rather than either.

**Tools.** Pass `deny_servers` to withhold whole MCP servers from the sub-agent (its tools, its resources, and its prompts) or `deny_tools` to withhold individual tools by name. Both union with whatever [`[subagents]`](../configuration/config-file.md#subagents) already denies; there is no way to grant something back, so a nested `agent_spawn` can only ever narrow further. Config is the place to put a restriction you always want, since the failure mode this guards against is an orchestrator forgetting to ask for it.

**Context is granted, not inherited.** A sub-agent starts with a clean slate and receives only what you ask for:

- `memory: "read"` grants read access to your memory store. Default `"none"`, because memories from unrelated work are context the worker pays for and reasons from. Sub-agents can never write to the store — record anything worth keeping yourself, from the worker's report.
- `instructions: "inherit"` hands over your [instructions file](../usage/instructions.md) verbatim. Default `"none"`, because those instructions describe *you*: your persona, how to address the user, what to volunteer. A worker handed one task by one of your turns is not you. Grant them when the task needs the project's standing rules and quoting the relevant ones into `prompt` would be lossy or expensive; pass a `skill` when the direction is reusable.

Neither can be granted beyond what you hold yourself, so authority only narrows going down a chain of sub-agents. A worker you gave no memory cannot give its own worker any.

**Follow-up.** `agent_spawn` returns the sub-agent's id on the first line of its result, above the report. Keep it if you might have a second question: with it you can call `agent_followup` instead of re-spawning a worker that would have to rediscover everything.

## `agent_list` / `agent_followup` / `agent_delete`

A sub-agent is not a one-shot. Its conversation persists under its own session, so you can go back to it.

- **`agent_list`** — the sub-agents this session spawned, one per line as `<id>\t<cwd>\tturns=<n>\tlast_active=<timestamp>`. Direct children only: a worker's own sub-agents belong to it and appear in *its* list.
- **`agent_followup({agent, prompt, scratchpad?})`** — asks a sub-agent another question. It still has its own conversation, so it can build on what it already found rather than starting from your summary of it. Returns its new report.
- **`agent_delete({agent})`** — discards a sub-agent: its conversation, its scratchpad entries, and any sub-agents it spawned in turn. Nothing it wrote to disk is touched. Worth doing once you have what you needed, so a long session isn't carrying every worker it ever ran.

All three refuse an id that isn't a child of the current session, so one session can never drive or delete another's workers.

**A follow-up runs under the terms of the spawn, not your current ones.** The permission level, the deny lists, the memory level and the inherited scratchpad names are recorded when the sub-agent is created and replayed on every follow-up. If you spawned a worker at `read` and have since switched to `unrestricted`, following up on it still runs it at `read`. That is deliberate: otherwise a second question would be a way to escalate a worker you deliberately restricted.

Two things do *not* survive a follow-up, because they only ever lived in memory: the sub-agent's todo list, and which files it had read. It is told as much at the start of the turn.

One follow-up at a time per sub-agent. A second concurrent call on the same worker is refused rather than interleaved, since both would be appending to one conversation from a view of it that the other has already changed.

## The `skill_*` tools

Skills are knowledge packages stored in `~/.config/meka/skills/<name>/SKILL.md`. The per-turn context lists the installed ones with their descriptions; these tools open, search, and (when enabled) maintain them.

- `skill_read({"name": "<skill-name>"})` returns the full body, prefixed with the skill's base directory.
- `skill_search({"pattern": "<regex>"})` matches each line of every skill, bodies included. This is what reaches skills the capped index did not list, and what answers "which of my skills covers this" when the one-line descriptions do not.
- `skill_write({"name": ..., "description": ..., "priority": ..., "body": ...})` creates or updates a skill. Omitting `body` keeps the existing one.
- `skill_delete({"name": ...})` removes the skill's whole directory, bundled files included.

The last two are registered only when [`[skills] agent_managed`](../configuration/config-file.md#skills) is on, and never for a sub-agent. See [Skills](../usage/skills.md) for how to author skills and [Letting the Agent Manage Skills](../usage/skills.md#letting-the-agent-manage-skills) for when to hand authoring to the agent.

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

## `conversation_search` / `conversation_read`

Search and re-read this session's **full** conversation, including earlier turns that [compaction](../usage/interactive-mode.md#compact) summarized and removed from the model's context. Compaction never deletes turns (it appends a boundary and hides the older ones); these tools read straight from the on-disk event log, so a detail the compaction summary dropped is still recoverable.

`conversation_search` searches and returns matching lines, each tagged with a message index (`#N`) and role:

```text
conversation_search({"query": "auth token", "regex": false, "limit": 20})
```

- `query` (required) — text to search for; a literal substring (case-insensitive) unless `regex` is set.
- `regex` — treat `query` as a case-sensitive regular expression. Default: `false`.
- `limit` — maximum matches to return (capped at 100). Default: 20.

`conversation_read` reads turns by the `#N` index that `conversation_search` reports:

```text
conversation_read({"start": 47, "count": 3})
```

- `start` (required) — 1-based message index to read from.
- `count` — number of consecutive messages to read (max 20). Default: 1.
- `scratchpad` — save the output to a scratchpad entry instead of returning it inline.

After a compaction, the summary message reminds the agent that these tools exist. Large tool outputs appear as `<large-output>` references in both `conversation_search` and `conversation_read` results (rather than inlining the full payload); read their full content with `scratchpad_read`.

## `context_check` / `context_compact`

Where `conversation_*` reads the **archive** (the full log on disk, including turns compaction removed from the window entirely), `context_*` manages the **live window**.

`context_check` takes no arguments and reports the current state:

```text
Using 84000 of 200000 tokens (42%).
Headroom: 76000 tokens before auto-compaction fires at 80%.
Kept verbatim on compaction: about 16000 tokens of the most recent turns; everything
older is replaced by a summary.
Fixed overhead: about 12000 tokens of system prompt and tool schemas (estimated).
Compaction does not reclaim this.
Conversation: about 72000 tokens, which is the part compaction acts on.
Compactions so far: none, so nothing has been summarized away yet.
```

This exists because the pushed `[Context budget]` block is rendered once, at the start of a turn, and so does not move while the agent works. During a long tool loop it is stale. See [What the Agent Sees](../usage/sessions.md#what-the-agent-sees).

`context_compact` requests a compaction at the end of the current turn:

- `instructions` — what to preserve or drop. Takes precedence over the default summary sections.
- `keep_recent` — whether to keep the most recent turns verbatim. Default `true`; `false` starts clean.

There is a third tool, `context_replace`, that exists only inside a checkpoint turn and is how the agent submits its summary. It is deliberately absent from the ordinary catalogue and from `[tools]` configuration. See [Compacting a Session](../usage/sessions.md#compacting-a-session).

## Redirecting output to the scratchpad

Several tools (`execute_command`, `find_files`, `search_contents`, `fetch_url`, `agent_spawn`) accept an optional `scratchpad` parameter that redirects their output to a named scratchpad entry instead of returning it inline. When this parameter is set, the tool produces its **full, untruncated output**: internal result-count caps (`find_files` 500, `search_contents` 100) and length caps (`fetch_url` `max_length`) are lifted for the scratchpad-bound result.
