# Scheduling

Scheduling lets the agent arrange its own future turns. Without it, meka only ever acts when
something outside it asks: a human typing, an editor sending a prompt, a client calling the HTTP
API. A scheduled job is the one trigger nothing else supplies, which is what makes meka usable as a
daemon or a standing assistant rather than a tool you drive.

The agent creates jobs itself through the `schedule_create` tool, so scheduling is usually a
conversation:

> **You:** remind me in 20 minutes to check the deploy
>
> **meka:** Created job `7f3a1b2c` (once at 2026-08-11 15:22 CEST). I'll remind you then.

## What a job is

A job pairs a **schedule** with a **prompt**. When it fires, the prompt is delivered as a turn.

| Schedule | Meaning | Example |
|----------|---------|---------|
| `at` | Fire once, then delete itself | `20m`, `2h`, `2026-08-12T09:00:00Z` |
| `every` | Fire on a fixed interval | `30m`, `1h`, `1d` |
| `cron` | Fire on a 5-field cron expression, in local time | `0 9 * * 1-5` |

Durations use the same syntax as `config.toml`, so `every = "30m"` means what
`[serve] idle_timeout = "30m"` means. Two things to know about it: **`m` is minutes and `M` is
months**, and decimals work (`1.5h` and `1h 30m` are the same duration).

Cron expressions have **no seconds field**, and follow standard Vixie semantics: when both
day-of-month and day-of-week are set, the job fires when **either** matches, not both.

## Gates: watching something without burning tokens

A plain recurring job spends a full model turn on every fire, whether or not anything happened.
Checking something every 15 minutes is roughly a hundred turns a day to say "nothing new" ninety-odd
times.

A **gate** is a shell command run before the turn. Only if it says something happened does the turn
occur. The interval then costs a process spawn instead of a model call, which is what makes a short
cadence reasonable:

```
schedule_create(
  every: "30s",
  gate: { command: "gh pr checks 123 --json state -q '.[].state' | sort -u", fire: "on-change" },
  prompt: "CI state for PR 123 changed. Investigate and report."
)
```

| `fire` | Fires when | Use for |
|--------|-----------|---------|
| `on-change` (default) | stdout differs from the previous run | "tell me when the build **finishes**" |
| `on-success` | the command exits 0 | "is this true yet" |

The gate's stdout is passed into the turn it triggers, so the model does not re-run the check the
gate just ran.

A gate's first evaluation always fires: with no previous output to compare against, "changed" is the
honest answer, and it means a typo in the command surfaces immediately instead of lying quiet.

> **Gates need `write` permission.** A gate is a shell command that runs unattended, on a timer,
> until someone cancels it: a longer-lived grant than `execute_command`, which at least ends with
> the turn that called it. Ungated reminders work at `read`.

A gate that fails or times out is **not** treated as "nothing happened". It is logged as a warning,
because a watcher whose command broke otherwise looks exactly like a healthy watcher with nothing to
report.

## Where jobs run

Jobs belong to the session that created them, and only fire while that session is live in some meka
process. That makes the two hosts behave differently, and the difference is worth knowing before you
rely on one:

| Host | Fires | Notes |
|------|-------|-------|
| `meka serve` | Every job, always | Revives evicted sessions on demand. **The durable path.** |
| REPL | Only jobs for the session it has open | Best-effort; a job goes dormant if you next start a different session |
| ACP | **Never** | Can create jobs, but runs no scheduler of its own |
| `--oneshot` | Never | The process exits; jobs stay on disk for a later run |

If you want a job to fire reliably whether or not you are sitting at a terminal, run `meka serve`.
In the REPL, a job created in one session resumes only when that session does; `meka --continue`
picks up where you left off.

Jobs all live in the same database, so a host that does not fire a job has not lost it. A job
created from an editor over ACP fires as soon as something that *does* run a scheduler picks it up,
which in practice means a `meka serve` daemon pointed at the same data directory. Without one, an
ACP-created job simply waits.

When a job fires at an idle REPL prompt, the turn interrupts the prompt and runs exactly like one
you typed: output streams, Ctrl+C interrupts it, and anything you had half-typed is handed back
afterwards.

## Restarts and missed jobs

Jobs live in meka's database, so restarting the process (or the `meka serve` systemd unit) does not
lose them. What happens to jobs whose time passed while meka was down depends on the kind:

- **Recurring jobs fire once and resume.** A 30-second job that was down for six hours has 720
  missed occurrences; it produces exactly one turn, which is told how many it stands in for. It is
  then rescheduled from now, so an outage never turns into a burst.
- **One-shot jobs fire if they are still relevant.** Past `[schedule] missed_grace` (24 hours by
  default) they are dropped instead. A reminder to join a standup, delivered five days late, is
  noise. One that does fire is told how late it is, so the agent can judge whether it still matters.

## Isolated jobs

By default a job's turn joins the conversation that created it, so you come back to a session
containing what happened while you were away.

Pass `isolated: true` and the turn runs in a fresh session instead. Much cheaper for anything
recurring, because the creating conversation's history is not replayed on every fire, but the
result does not appear in that conversation. Isolated runs are ordinary sessions, so
`meka session list` and `meka session export` can see them.

**Only `meka serve` honours `isolated`.** The REPL has one agent and one conversation, so a job that
asked for isolation runs in the current conversation there instead, with a warning saying so.

## Unattended turns and permissions

A scheduled turn has no human on the other end. In `ask` permission mode, every approval prompt
therefore resolves to **deny**, and the job fails to do whatever needed approval. The denial appears
in the session transcript rather than anywhere louder, so a job created in `ask` mode that seems to
do nothing is worth checking there first. Jobs created in `read` or `write` mode run at that level
as normal.

## Inspecting jobs

The agent sees a short index of the current session's jobs in its per-turn context, so it can avoid
scheduling a duplicate. For details it calls `schedule_list`.

From your side:

```bash
meka schedule list                  # every session's jobs
meka schedule list --session <uuid> # one session
meka schedule cancel 7f3a1b2c       # by id, or any unique prefix
```

In the REPL, `/schedule` lists the current session's jobs and `/schedule cancel <id>` cancels one.

## Configuration

```toml
[schedule]
enabled = true          # default true; false hides the tools and stops the scheduler
poll_interval = "10s"   # how often due jobs are checked
missed_grace = "24h"    # how late a one-shot may be and still fire
gate_timeout = "30s"    # wall-clock budget for a gate command
max_jobs = 50           # per-session ceiling, refused at schedule_create
```

`poll_interval` is the real resolution floor: a job with a shorter interval fires once per tick, not
once per interval.

Setting `enabled = false` keeps the three `schedule_*` tool schemas out of every request and leaves
existing jobs on disk without firing.

## Tips

- Write the prompt for a reader who has no context. The conversation that created the job may be
  long over, and for an isolated job it was never there at all.
- Prefer `isolated: true` for anything recurring; the token difference is large.
- Reach for a gate whenever the answer is usually "nothing happened".
- Keep gate commands fast and read-only. They run on every tick, and a gate that changes something
  is a side effect on a timer.
- If a schedule matters, check what `schedule_create` reports back: it states the resolved next fire
  in absolute local time, which is how you catch a cron expression that parsed fine and means
  something other than you intended.
