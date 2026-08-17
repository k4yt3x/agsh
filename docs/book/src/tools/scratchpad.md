# Scratchpad

The scratchpad is a session-scoped working memory that the agent can use to store, retrieve, edit, and manage content without consuming conversation context. Entries are identified by string names and persist across turns within a session.

## When the Scratchpad is Used

- **Proactively**: The agent stores intermediate results (extracted text, API responses, research notes) for later use.
- **Via `scratchpad` parameter**: Any tool can save its output directly to the scratchpad by including a `scratchpad` parameter in the tool call.
- **Automatically**: When a tool's output exceeds 30,000 characters, it is saved to the scratchpad under an auto-generated name (e.g., `execute_command_1`) and replaced with a preview in the conversation.

## Tools

All five tools below ship default-active; no `load_tool` round-trip is required to use any of them.

### `scratchpad_write`

Store content in the scratchpad. If the name already exists, the content is overwritten.

**Permission:** Read

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `name` | string | yes | Name for the entry |
| `content` | string | yes | The content to store |

### `scratchpad_read`

Read or search a scratchpad entry by name.

**Permission:** Read

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `name` | string | yes | The entry name |
| `offset` | integer | no | Character offset to start reading from (default: 0) |
| `limit` | integer | no | Maximum characters to return; no hard cap. Pass the entry's `size` to load all content in one call. (Default and exact value are advertised in the tool's parameter schema.) |
| `regex` | string | no | Search the entry and return matching lines (capped, exact value advertised in the tool's parameter schema). |

### `scratchpad_edit`

Edit a scratchpad entry in place. Provide `content` for a full overwrite, or `old_string`/`new_string` for targeted replacement.

**Permission:** Read

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `name` | string | yes | The entry name |
| `content` | string | no | Full replacement (mutually exclusive with old/new) |
| `old_string` | string | no | String to find |
| `new_string` | string | no | Replacement string |
| `replace_all` | boolean | no | Replace all occurrences (default: false) |

### `scratchpad_list`

List all scratchpad entries with their name, size, and creation time. No parameters.

**Permission:** Read

### `scratchpad_delete`

Delete a scratchpad entry by name.

**Permission:** Read

| Name | Type | Required | Description |
|------|------|----------|-------------|
| `name` | string | yes | The entry name to delete |

## Handing entries to a sub-agent

`agent_spawn`'s `inherit_scratchpad` takes a list of the parent's entry names and grants the
sub-agent read-only access to exactly those:

```text
agent_spawn(prompt: "summarise the failures", inherit_scratchpad: ["build_log"])
```

The sub-agent's `scratchpad_read` falls back to the parent for an inherited name, and its
`scratchpad_list` shows the entry with origin `inherited`. `scratchpad_write`, `scratchpad_edit` and
`scratchpad_delete` targeting one return an error, so a worker cannot rewrite what it was lent.

This is how a large captured output reaches a sub-agent without being re-inlined into the prompt.
When you expect to delegate a result later, name it at the source with the `scratchpad` parameter
(`execute_command({command: "...", scratchpad: "build_log"})`) so there is a semantic name to pass
through.

## Lifecycle

- Entries are scoped to the session and persist across turns.
- Entries survive session compaction (`/compact`).
- Entries are deleted when the session is deleted.
- Two sessions can have entries with the same name without conflict.
- Writing to an existing name overwrites it silently.
