# Background Tasks

An ordinary tool call holds the turn open until it returns. That is right for reading a file and wrong for a twenty-minute build: the agent cannot answer anything else while it waits, and the alternative it reaches for on its own, `nohup … &` plus polling, gets no notification when the work is done.

A **background tool call** returns immediately with a task id and delivers its result later, as its own turn.

**Off by default.** Turn it on with:

```toml
[background]
enabled = true
max_tasks = 10   # concurrent per session
```

## When to enable it

This changes the contract of the primary interaction. Without it, you ask and the agent answers. With it, you ask, the agent answers, and something else may interrupt you several minutes later.

That is right for an **assistant** that runs unattended, keeps talking while work proceeds, and reports when it lands. It is usually wrong for the interactive case, someone at a terminal using the REPL like a command line, where blocking is what you want and asynchrony is a surprise.

Every other capability block (`[schedule]`, `[skills]`, `[memory]`) defaults on. This one does not, because those add capability without changing when a turn ends.

## How the agent uses it

Once enabled, every tool gains an optional `background` parameter, including tools from MCP servers, since a slow MCP call is exactly the kind worth detaching:

```text
execute_command({"command": "cargo test --all", "background": true})
```

That returns something like:

```text
Started in the background as task 7f3a1c22 (cargo test --all). It is still
running; its result will be delivered to you when it finishes.
```

The agent then carries on. When the task ends, its outcome arrives as a new turn:

```text
[Background task reporting at 2026-08-12 14:31 CEST]

7f3a1c22 (cargo test --all) finished after 12m 4s.

test result: ok. 1674 passed; 0 failed
```

Running tasks also appear in the per-turn context under `[Background]`, so the agent can see what it already started and does not launch a second copy:

```text
[Background]
Tasks you started and did not wait for, still running. Each will report to you
on its own when it finishes; do not poll for them and do not start a second
copy of work already listed here.

- **7f3a1c22**: cargo test --all
```

That section is rendered fresh every turn from live state, like `[Todo list]`, so it is always current rather than something the agent has to reconstruct. It carries **no results**: an outcome is permanent and belongs in the conversation, delivered as its own turn. The section disappears entirely when nothing is running.

## Outcomes

Every task ends in one of four states, and every one of them is reported (as a turn everywhere except `--oneshot`, which has no later turn and prints them instead):

| Status | Meaning |
|--------|---------|
| `completed` | The tool returned successfully |
| `failed` | The tool returned an error |
| `cancelled` | Stopped on request, via `task_cancel`, `/tasks cancel`, or a second Ctrl+C |
| `interrupted` | The process holding it went away |

`interrupted` is the one that matters most. A task in flight when meka exits cannot be resumed, so it is retired and reported the next time something takes ownership of that session: a REPL resume, a `meka serve` reattach, or an ACP `session/load`. Nothing is written at exit; the *next* owner does the retiring, because holding the session lock is what proves the previous owner is gone. Without this the agent would wait forever on a result it had usually already promised someone.

Large output is written to a [scratchpad](../tools/scratchpad.md) entry and the delivered turn carries the beginning plus the entry name, so a long build log does not occupy the conversation permanently.

## Managing tasks

The agent has `task_list` and `task_cancel`. You have:

```bash
/tasks                    # list this session's tasks
/tasks cancel 7f3a1c22    # stop one
/tasks cancel --all       # stop all of them
```

A cancelled task still reports back, so the agent learns it stopped rather than waiting on it.

## Ctrl+C

**The first Ctrl+C cancels the turn only. Background tasks keep running.**

This is the shell's contract, where Ctrl+C signals the foreground process group and `&`-ed jobs survive. Losing a twenty-minute build to a keystroke aimed at the answer on screen is unrecoverable, and it is not what the keystroke meant.

meka prints what survived so nothing is hidden:

```text
Interrupted.
2 background task(s) still running. Press Ctrl+C again during a turn to stop
them, or use /tasks.
```

A **second** Ctrl+C during the same turn stops them. Between turns, `/tasks cancel --all` is the route.

## Where it works

| Host | Behaviour |
|------|-----------|
| REPL | Full. Outcomes arrive between turns |
| `meka serve` | Full, for sessions currently resident |
| ACP | Full, for sessions the editor has open |
| `--oneshot` | The run waits for outstanding tasks before exiting |

A one-shot run exits with the turn, so there is no later turn to deliver into. Rather than kill the work halfway through, it waits for every outstanding task and then prints the outcomes on stderr. The agent does not see them: its turn is already over. So a background call under `--oneshot` costs the same wall-clock as a synchronous one without the result reaching the model, which makes it worth avoiding rather than a feature to reach for.

Sub-agents (`agent_spawn`) deliberately cannot start background tasks. A sub-agent's session ends with the single turn that spawned it, so it could neither outlive that turn nor be around to hear the result.

## Concurrent edits

Background tasks make it ordinary for two agents to work in one directory at once. meka does not lock anything: coordination is the orchestrating agent's job, exactly as it is between two people on one machine.

What it does do is make a lost race **visible**. `edit_file` records what a file looked like when it was read, and refuses an edit against a file that changed since:

```text
Error: file 'src/main.rs' changed on disk after you read it. Something else
wrote to it (a shell command, another agent, or the user). Read it again
before editing so you are not overwriting that change, or set force=true to
edit anyway.
```

This applies whether or not background tasks are enabled: a shell `sed -i`, or your own editor, produces the same situation.

A file served by the editor under [ACP](./acp.md) is checked against the editor rather than the disk,
since the bytes the agent saw were the editor's. The check is the same; only the thing it compares
against changes. See [read-before-edit](../tools/file-operations.md#edit_file).
