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
day-of-month and day-of-week are set, the job fires when **either** matches, not both. A six-field
expression is rejected rather than read as Quartz, where `*/10 * * * * *` would mean every ten
seconds instead of the every-ten-minutes it looks like.

A pattern that matches no calendar date (`0 0 30 2 *`) is rejected when the job is created. One whose
next occurrence is far off is not: `0 0 29 2 *` waits up to four years for the next February 29th and
stays on the books until then.

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

**An `on-change` gate is only as good as the stability of its output.** The command should print
something that changes when, and only when, the watched thing does, which is a stronger requirement
than "read-only" and is where most gates go wrong. It fails in both directions. Output carrying
something that moves on its own (a timestamp, an elapsed time, an unsorted list whose order varies)
differs on every evaluation, so the gate fires every tick and costs more than the ungated job it
replaced. Output that can return to an earlier value between polls (a bare count, where two events
arrive and one is consumed) reads as unchanged, and the gate silently misses what happened in
between. Pairing a count with a monotonic marker, as in `git rev-list --count HEAD` alongside the
commit sha, avoids both.

> **Gates need `write` permission.** A gate is a shell command that runs unattended, on a timer,
> until someone cancels it: a longer-lived grant than `execute_command`, which at least ends with
> the turn that called it. Ungated reminders work at `read`.

A gate that cannot run at all -- it times out, or the shell fails to start it -- is **not** treated
as "nothing happened". It is logged as a warning, because a watcher whose command broke otherwise
looks exactly like a healthy watcher with nothing to report.

A non-zero **exit code** is different, and `on-change` deliberately does not treat it as a failure:
for a large class of good gates it is the signal. `diff -q a b` and `git diff --exit-code` exit 1
exactly when there *is* a difference; `grep ERROR log` exits 1 through the entire quiet period it is
watching; `curl -f` exits non-zero until the endpoint returns. `on-change` compares stdout as usual
and logs the exit code, so a genuinely broken command is visible in the log without a working one
being silenced. `on-success` reads the exit code by definition.

## Where jobs run

Jobs belong to the session that created them, and only fire while that session is live in some meka
process. That makes the two hosts behave differently, and the difference is worth knowing before you
rely on one:

| Host | Fires | Notes |
|------|-------|-------|
| `meka serve` | Every job, except on a session another process has locked | Revives evicted sessions on demand. **The durable path.** |
| REPL | Only jobs for the session it has open | Best-effort; a job goes dormant if you next start a different session |
| ACP | Only jobs for sessions the editor has open | The prompt appears in the transcript as the turn that triggered the reply |
| `--oneshot` | Never | The process exits; jobs stay on disk for a later run |

If you want a job to fire reliably whether or not you are sitting at a terminal, run `meka serve`.
In the REPL, a job created in one session resumes only when that session does; `meka --continue`
picks up where you left off.

Jobs all live in the same database, so a host that does not fire a job has not lost it. A job whose
session nobody has open simply waits, and fires as soon as something that can run it picks it up:
another host, or a `meka serve` daemon pointed at the same data directory.

Two hosts sharing a session do not fight over its jobs. A session is held by one process at a time,
and `meka serve` leaves that session's jobs to whoever holds it rather than reaching for them and
handing the occurrence back afterwards — which matters most for a gated job, since deciding late
would mean running its shell command on every tick.

The exception is an `isolated` job, which runs in a session of its own and so needs nothing from the
one that created it. `meka serve` stays eligible for those whatever else holds the session, because
it is the only host that honours the flag. If you have a REPL open on the same session at the same
time, though, either host may take a given occurrence, and the REPL will run it in your conversation
with a warning. Run isolated jobs under `meka serve` alone if you want the flag respected every time.

### More than one host on the same database

Several meka processes pointed at one data directory — two `meka serve` instances, or a daemon and
a terminal — all poll the same table, so the same occurrence appears in several due lists at once.
Each occurrence is nevertheless run once. A host takes it by moving the job's next fire time off
the value it read, in a single conditional write, and the hosts that lose that write stop before
evaluating the gate: no duplicate shell command, no duplicate turn, no duplicate isolated session.
Which host wins is a race between their tickers and is not something you can pin down; that only
one wins is.

Under ACP the editor is a live client, which changes one thing: `ask`-mode approvals genuinely
round-trip, so a scheduled job can prompt you in the editor rather than being denied for want of
anybody to ask. Stopping a scheduled turn works the same as stopping any other.

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

That collapsing is per job. A session with several jobs all due at once still wakes to a turn each,
and a sweep runs at most `[schedule] max_consecutive_fires` (5 by default) of any **one session's**
jobs before moving on. The rest keep their occurrence and their gate baseline and are taken by the
next sweep, most-overdue first, so nothing is lost and nothing starves. A job held over runs no gate,
so holding one over is nearly free — the sweep still evaluates whether the job is one it can run.

**What this does and does not do.** It bounds a *batch*, not a total: forty due jobs still produce
forty turns, and they are not spaced out — a sweep that ran long leaves the next one already due.
What changes is that they arrive in groups of five, so under `meka serve` another session's single
due job is reached after five of the first session's rather than after all forty.

If you want a large backlog not to land at all, that is not what this setting is for. Cancel the
jobs (`meka schedule list`, then `meka schedule cancel <id>`) before starting a host that will fire
them, or leave `[schedule] enabled = false` while you clear it.

A **recurring** job that fires and then fails — most often because the provider is unreachable —
leaves nothing behind in the conversation: its prompt is withdrawn, because the job produces it again
on the next occurrence. Without that, an outage would deposit one unanswered message per fire for as
long as it lasted. A **one-shot** keeps its prompt, because nothing will produce it again — its row
is already gone by the time the turn runs, so that message is the last trace the reminder ever fired.
A turn that got as far as running a tool keeps everything either way, since there is real work behind
it. Failures are recorded regardless: `meka serve` logs them and sends a `schedule.fired` webhook
with `status: "failed"`.

## Isolated jobs

By default a job's turn joins the conversation that created it, so you come back to a session
containing what happened while you were away.

Pass `isolated: true` and the turn runs in a fresh session instead. Isolated runs are ordinary
sessions, so `meka session list` and `meka session export` can see them.

The trade is cost against continuity, and neither side is the default answer. Isolation is cheaper
for anything recurring, because the creating conversation's history is not replayed on every fire,
and it keeps a job firing every two minutes from filling the conversation you actually talk in with
its own results. But an isolated turn remembers nothing said in the parent, so a job whose value
depends on that memory (a look-back at something the agent itself did, a watcher that should notice
it has already reported this) is worse in isolation no matter how well the prompt is written. Reach
for it when the job is self-contained, and leave it off when it is not.

**Only `meka serve` honours `isolated`.** The REPL and ACP each drive one conversation per session,
so a job that asked for isolation runs in that conversation instead, with a warning saying so.

## Unattended turns and permissions

Under `meka serve` a scheduled turn has no human on the other end, so in `ask` permission mode every
approval resolves to **deny** and the job fails to do whatever needed approval. The denial appears in
the session transcript rather than anywhere louder, so a job in `ask` mode that seems to do nothing
is worth checking there first.

The REPL and ACP both have someone attached, so approvals reach them normally.

A job's **turn** runs at whatever permission the session holds when it fires, not at the level the
job was created with: the level lives on the session and the session is mutable, through Shift+Tab in
the REPL or `PATCH /v1/sessions/{id}` under `serve`.

A job's **gate** is the exception, because it is a shell command that runs unattended and
unsandboxed. It needs `write` from two places every time it comes due: the level recorded on the job
when it was authored, and the level in force *now* -- the session's, or the host process's for a
session that carries none. Drop the session to `read`, or restart `meka serve --permission read` so
it inherits the job, and the gate stops running -- and with it the job, because a gate is the
condition on the job and an unevaluated condition has not been met. The occurrence is declined the
same way a gate that ran and said "nothing happened" declines it, and a warning is logged naming the
job. Raise the session back to `write` to restore it.

Firing the reminder ungated instead would be the more forgiving-looking choice and the wrong one: it
turns a conditional job into an unconditional one, so an `every = "1m"` watcher that normally speaks
once a week would deliver a turn a minute for as long as the session stayed below `write`. An
*ungated* job is unaffected by any of this and keeps firing.

Both halves are load-bearing. The recorded level alone can never refuse anything, since creating a
gate already demands `write`; the live level is what makes a withdrawal real. The recorded level
still matters for a job created before it was stored, which reads as "no authority" and stays
refused.

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
max_consecutive_fires = 5 # per-session ceiling on turns spent in one sweep
```

`max_consecutive_fires` bounds a batch, not a total. A sweep contains its turns, so lowering it does
not stop a backlog landing, nor slow it down — it splits it into smaller groups with other sessions
interleaved between them. Raising it above the number of jobs one session can have due at once has
no effect at all.

With a long `poll_interval` and a small budget, a large backlog can take long enough to drain that a
one-shot job ages past `missed_grace` and is dropped (with a warning) before its turn comes.

`poll_interval` is the real resolution floor: a job with a shorter interval fires once per tick, not
once per interval.

Setting `enabled = false` keeps the three `schedule_*` tool schemas out of every request and leaves
existing jobs on disk without firing. `POST /v1/sessions/{id}/schedule` refuses with a 422 rather
than accepting a job that could never run; `GET /v1/schedule` and `DELETE /v1/schedule/{job_id}`
keep working, so jobs left over from before the flag was flipped can still be listed and cleared.

## Tips

- Write the prompt for a reader who has no context. The conversation that created the job may be
  long over, and for an isolated job it was never there at all.
- Reach for `isolated: true` when a recurring job is self-contained; the token difference is large.
  Leave it off when the job needs to remember what happened in the conversation that created it.
- Reach for a gate whenever the answer is usually "nothing happened".
- Keep gate commands fast and read-only. They run on every tick, and a gate that changes something
  is a side effect on a timer.
- Check what a gate's command prints across two runs where nothing happened. Identical output is the
  whole mechanism, and anything varying inside it turns the gate into a timer.
- If a schedule matters, check what `schedule_create` reports back: it states the resolved next fire
  in absolute local time, which is how you catch a cron expression that parsed fine and means
  something other than you intended.
