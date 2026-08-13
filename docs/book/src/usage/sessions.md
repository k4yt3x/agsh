# Sessions

Sessions persist your conversation so you can resume later. Each session is identified by a UUID and stored in a SQLite database.

## How Sessions Work

- A session is **not** created when meka starts. It is created lazily when you send the first message.
- When a session is created, its UUID is printed to stderr.
- When you exit meka (Ctrl+D), the session UUID is printed again so you can note it for later.
- Sessions include the full conversation: your inputs, the agent's responses, and tool call results.

## Resuming a Session

### Continue Last Session

```bash
meka -c
```

This resumes the most recently updated session. `-c` takes no value, so you can follow it with an opening prompt: `meka -c "and now add tests"`.

### By UUID

```bash
meka -r 550e8400-e29b-41d4-a716-446655440000
```

The agent loads the previous conversation and continues from where you left off.

### By UUID Prefix

If the value passed to `-r` isn't a valid UUID, meka treats it as a leading prefix and looks up sessions whose ID starts with it. This avoids having to copy the entire UUID:

```bash
meka -r 550e            # works if exactly one session starts with `550e`
meka -r 5               # likely ambiguous; meka lists matching IDs and exits
```

When a prefix matches multiple sessions, meka prints the matching IDs (most-recent first) so you can disambiguate. Type a few more characters until the prefix is unique.

### What a Resume Does Not Restore

A resume restores the conversation, not the world it ran in. The messages come back verbatim, which means the agent reads its own earlier tool calls and can reasonably assume their effects still hold. Two kinds of state do not survive the process that made them:

- **Which files have been read.** meka tracks reads in memory so `edit_file` can refuse to write over a file the agent has not seen. A new process starts with that record empty, so the first edit to any file asks for a `read_file` first.
- **Anything an MCP server was holding.** A loaded database, an authenticated session, a subscription — these belong to the server's process, not to the conversation, and a reconnect drops them. meka has no way to model what a given server keeps open.

Everything else is restated in the per-turn context on every turn regardless (permission level, working directory, todo list, tool catalogue), and background tasks that were running deliver an `interrupted` outcome, so none of those can go stale unnoticed.

Because the second kind is unknowable from meka's side, the first turn after a resume carries a `[Session resumed]` note telling the agent to re-establish rather than assume. It appears once and is not repeated. There is nothing to configure.

## Session Locking

Only one meka instance can be attached to a session at a time. This prevents race conditions from concurrent writes.

- If you try to resume a session that is locked by a running meka process, you will get an error.
- If the locking process has exited (crashed or was killed), meka detects this and allows you to take over the lock.
- Under ACP (`meka acp`), the lock is released as soon as the editor disconnects: closing the connection (stdin EOF) or sending SIGTERM/Ctrl-C makes `meka acp` exit, so the session can be reopened immediately.

## Storage Location

Sessions are stored in a SQLite database at a platform-specific location:

| Platform | Path |
|----------|------|
| Linux | `~/.local/share/meka/meka.db` (`$XDG_DATA_HOME/meka/meka.db`) |
| macOS | `~/Library/Application Support/meka/meka.db` |
| Windows | `%APPDATA%\meka\meka.db` |

## Database Schema

The database has three tables:

**sessions**, one row per session:

| Column | Type | Description |
|--------|------|-------------|
| `id` | TEXT (UUID) | Primary key |
| `created_at` | TEXT (RFC 3339) | When the session was created |
| `updated_at` | TEXT (RFC 3339) | When the session was last updated |
| `locked_by` | TEXT (PID) | PID of the process holding the lock, or NULL |
| `metadata` | TEXT | Reserved for future use |

**messages**, one row per message in a session:

| Column | Type | Description |
|--------|------|-------------|
| `id` | INTEGER | Auto-incrementing primary key |
| `session_id` | TEXT (UUID) | Foreign key to `sessions.id` |
| `role` | TEXT | `user`, `assistant`, or `tool_results` |
| `content` | TEXT | Message content (plain text or JSON) |
| `created_at` | TEXT (RFC 3339) | When the message was saved |

**tool_outputs**, scratchpad entries, one row per entry:

| Column | Type | Description |
|--------|------|-------------|
| `session_id` | TEXT (UUID) | Part of composite primary key |
| `name` | TEXT | Part of composite primary key |
| `content` | TEXT | The stored content |
| `created_at` | TEXT (RFC 3339) | When the entry was created |

Scratchpad entries are scoped to a session. Two sessions can have entries with the same name. Entries are preserved across compaction but deleted when a session is deleted.

## History Retention

**meka never deletes sessions unless you ask it to.** Conversation history isn't reproducible, so there is no default cleanup by age and none at all by size.

If you do want a time window, set it explicitly:

```toml
[session]
retention_days = 30   # delete sessions not updated in 30 days, at startup
```

With that set, meka deletes matching sessions when the agent starts and says so at `warn` level, so a deletion you configured is still a deletion you see. Unset (the default) keeps everything forever.

To prune on demand instead, delete on your own schedule:

```bash
meka session delete --older-than-days 90   # same window, run when you choose
meka session delete <id> [<id>…]           # specific sessions
meka session delete --all                  # everything
```

Deleting a session also removes its messages, scratchpad entries, and any sub-agent children.

See [Config File](../configuration/config-file.md#session) for details.

## Context Window Limiting

Long sessions can exceed the LLM's context window or become expensive. The `context_messages` setting (default: `200`) limits how many recent messages are sent to the API:

```toml
[session]
context_messages = 100
```

The full history remains in SQLite for resumption. Only the API payload is truncated. The truncation preserves tool call chains (it never splits a tool use from its result).

The tool catalogue and skill list travel in the conversation rather than the system prompt, so they are subject to this window too. meka tracks where it last stated them and restates them in full once that message scrolls out, which works out to roughly once per window. Setting `context_messages` very low therefore makes those restatements more frequent.

### Compacting a Session

If a session becomes too long, you can use the `/compact` command to have the LLM summarize the conversation and replace older messages with a structured summary. A token-budgeted tail of the most recent messages is preserved verbatim (snapped to a clean user-turn boundary so tool calls aren't split). The structured summary captures the primary task and current state, key files and decisions, errors and fixes, every distinct user request, the next step (quoted verbatim to avoid drift), and any security-relevant constraints the user stated (preserved verbatim so they keep applying after compaction).

Compaction preserves scratchpad entries and the todo list, and re-injects environment context so the agent isn't disoriented after compaction. The tool catalogue, skill list, and MCP server instructions are restated in full on the next turn too, since the messages that carried them may have been summarized away. The summary message ends with a directive to resume the work directly rather than narrate the summary. Tools that the model loaded via `load_tool` before compaction stay loaded after; the deferred-tool active set is snapshotted into the compaction boundary, so resumed sessions don't re-issue `load_tool` for tools they already used. If a detail was dropped, the model can `conversation_search` / `conversation_read` the full pre-compaction history, which stays on disk.

Internally, compaction does not delete pre-compaction rows from the database. It appends a `compact_boundary` row to the `messages` table; the materialized view is reconstructed from the event log, so the persisted log itself stays append-only.

### Auto-Compact

When `auto_compact` is enabled (default: `true`), meka automatically compacts the conversation when the input token count exceeds 80% of the context window. This runs between turns, not during tool loops. The check is both reactive (the previous turn's reported usage) and proactive (an estimate of the next request, so a turn whose own input jumps over the window is compacted before it is sent). As a last resort, if the provider still rejects a request for exceeding the context window, meka compacts once and retries the turn instead of failing.

```toml
[session]
auto_compact = true
context_window = 200000  # optional override
```

### What the Agent Sees

Once a turn has been measured, the per-turn context block carries a `[Context budget]` line reporting occupancy and the threshold compaction fires at:

```text
[Context budget]
Using ~84k of 200k tokens (42%). The conversation is summarised automatically at
80%, which loses detail, so prefer to finish or checkpoint work before then.
```

The agent is expected to budget its own reading and to decide when a task will fit, so it needs the same number the harness uses. Without it, those are guesses. The line is suppressed when the window is unknown, and on the first turn of a session, when there is no measurement yet rather than a genuine zero.

It rides the per-turn context block rather than the system prompt because it changes every turn and the system prompt is the cached prefix.

## Listing Sessions

To see past sessions:

```bash
meka session list
```

This shows a table with each session's ID, last update time, and a preview of the first message:

```
ID                                    Updated              Preview
550e8400-e29b-41d4-a716-446655440000  2026-03-14 12:00:00  How do I implement a binary search tree?
a1b2c3d4-e5f6-7890-abcd-ef1234567890  2026-03-13 09:30:00  Fix the login page CSS
```

By default the 20 most recent sessions are shown. Use `-n` to change:

```bash
meka session list -n 50
```

## Exporting a Session

You can export any session as a Markdown file:

```bash
meka session export 550e8400-e29b-41d4-a716-446655440000
```

This writes `session-550e8400-e29b-41d4-a716-446655440000.md` in the current directory with the full conversation. User and assistant messages are rendered as Markdown sections, while tool calls and results are wrapped in collapsible `<details>` blocks. The export always covers the **entire** session, including turns that were later hidden from the model by [compaction](interactive-mode.md#compact) (each compaction point is marked with its summary).

To write to a specific file:

```bash
meka session export 550e8400-e29b-41d4-a716-446655440000 -o conversation.md
```

To print to stdout (for piping):

```bash
meka session export 550e8400-e29b-41d4-a716-446655440000 -o -
```

### JSON (structured, round-trippable)

Pass `--format json` for a structured export instead of rendered Markdown:

```bash
meka session export 550e8400-e29b-41d4-a716-446655440000 --format json
```

This writes `session-<id>.json`, a lossless dump of the session's event log (including input images and compaction boundaries), its cumulative stats, and scratchpad entries. Unlike Markdown, a JSON export also includes any **sub-agent child sessions** spawned during the conversation, and it can be re-imported with `meka session import`. It deliberately contains **no credentials**: API keys and OAuth tokens live in separate tables and are never part of an export.

## Importing a Session

Recreate a session from a JSON export:

```bash
meka session import session-550e8400-e29b-41d4-a716-446655440000.json
```

meka assigns the imported session (and any sub-agent children) **new** UUIDs so they can't collide with existing sessions, then prints the new root session ID. Resume it like any other session:

```bash
meka -r <new-id>
```

Read from stdin with `-`:

```bash
cat session.json | meka session import -
```

The import preserves the full conversation, per-message timestamps, cumulative stats, and scratchpad entries. Because the provider and model are chosen at run time (not stored per session), a resumed import uses your currently-active provider.

`updated_at` is stamped to the import time rather than restored from the export, so that restoring an archive older than a configured `retention_days` window isn't undone by the retention sweep on the next launch. `created_at` still carries the original.

## Forking a Session

Branch off an existing conversation without disturbing it:

```bash
meka session fork 550e8400-e29b-41d4-a716-446655440000
```

The copy starts with the original's full conversation and continues from there under a new UUID, which is printed on stdout so it can be captured:

```bash
meka -r "$(meka session fork 550e8400-e29b-41d4-a716-446655440000)"
```

Use it to try a different direction from a known-good point, to run a throwaway question against a large accumulated context, or to keep a conversation you're about to compact.

What the copy carries: the full event log, scratchpad entries, working directory, permission level, additional workspace roots, and cumulative stats. What it does **not**: sub-agent child transcripts (the sub-agent's result already sits in the parent conversation as a tool result, so the copy is complete without them), and the timestamps, which are stamped fresh.

A fork records no link back to the session it came from; it is a top-level session like any other.

Forking copies what has been committed to the database, so forking a session with a turn in flight can capture that turn partially: the user message is persisted before the model is called, and each assistant round lands together with its tool results as it completes. The copy may therefore end mid-turn, with a user message that has no reply yet, or an assistant round that was not the last. Because each round and its tool results are written as one unit, the copy is never internally inconsistent, just short. Fork between turns if you want an exact copy.

The same operation is available from the REPL as `/fork`, which switches you into the copy and leaves the original where you branched; over HTTP as `POST /v1/sessions/{id}/fork`; and over ACP as `session/fork`.

### Fork or export/import?

Both produce a runnable copy under a new ID. Reach for `fork` to branch a conversation you're working on, and for `export` + `import` to move a session between machines or keep an archive. Export/import also copies sub-agent transcripts and preserves `created_at`, because an archive should restore whole.

## Rewinding a Session

Drop the most recent turns from a session:

```bash
meka session rewind 550e8400-e29b-41d4-a716-446655440000
meka session rewind 550e8400-e29b-41d4-a716-446655440000 -n 3
```

The cut lands on a turn boundary, so a tool call is never separated from its result, and nothing is deleted: the dropped turns stay in the event log and still appear in `meka session export`, marked at the point of the rewind. The model simply stops seeing them.

The command takes the session lock, so it refuses to run while a REPL, `meka serve`, or `meka acp` holds the session; that process has its own copy of the conversation in memory and would write over the rewind on its next turn. In the REPL use `/rewind` instead. Under ACP or the HTTP API there is no in-session equivalent, so close the session in the editor (or stop the server) and run this command.

Its main use is recovering a session a provider has started refusing. A provider validates the whole conversation on every request, so one piece of content it rejects fails every later turn too. meka repairs a rejection caused by content it added during the current turn, and repairs a mislabelled image on resume, but anything older than that needs rewinding past.

## Deleting Sessions

Delete specific sessions by UUID:

```bash
meka session delete 550e8400-e29b-41d4-a716-446655440000
```

Delete multiple sessions at once:

```bash
meka session delete 550e8400-e29b-41d4-a716-446655440000 a1b2c3d4-e5f6-7890-abcd-ef1234567890
```

Delete every session not updated in the last N days:

```bash
meka session delete --older-than-days 90
```

This is the manual counterpart to [`retention_days`](#history-retention). It can't be combined with UUIDs or `--all`, and `0` is refused: it would match everything.

Delete all sessions:

```bash
meka session delete --all
```

## Input History

Separate from your saved conversations, meka keeps a rolling history of the prompts you *type* at the REPL, so **Up-arrow** recall and **Ctrl+R** reverse-search work across runs (shell-style). This is distinct from a session (a stored conversation) and from the `/history` slash command (which reprints the current conversation).

List recent input-history entries (oldest first; `-n 0` shows all):

```bash
meka history list
meka history list -n 100
```

Clear it entirely:

```bash
meka history clear
```

## Managing Sessions via SQLite

You can also manage sessions directly through the SQLite database. For example, to list all sessions:

```bash
sqlite3 ~/.local/share/meka/meka.db \
  "SELECT id, created_at, updated_at FROM sessions ORDER BY updated_at DESC;"
```
