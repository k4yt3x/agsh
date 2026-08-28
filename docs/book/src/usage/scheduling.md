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

A **gate** is a cheap check run before the turn. Only if it says something happened does the turn
occur. The interval then costs a tool call or a process spawn instead of a model call, which is what
makes a short cadence reasonable.

A gate has two halves. `check` is what to run, and `when` is what counts as "something happened":

```
schedule_create(
  every: "30s",
  gate: {
    check: { command: "gh pr checks 123 --json state -q '.[].state' | sort -u" },
    when: "changed"
  },
  prompt: "CI state for PR 123 changed. Investigate and report."
)
```

### What a gate can check

| `check` | Runs | Needs |
|---------|------|-------|
| `{ command: "..." }` | a shell command, unsandboxed | `unrestricted` |
| `{ tool: "name", arguments: {...} }` | a tool call, by the name the model uses | `read`, and the tool must resolve to `read` |

A tool gate is the one to reach for when a tool exists for the job. It is available at a far lower
permission, because a structured call to a server you configured is not a shell, and it returns
structured data that `when.at` can point into:

```
schedule_create(
  every: "1m",
  gate: {
    check: { tool: "mcp__mekabridge__unseen", arguments: {} },
    when: { at: "/chats", is: "not-empty" }
  },
  prompt: "There are unseen chats. Read them and reply if anything needs an answer."
)
```

**A gate may only call a tool that resolves to `read`.** A gate asks a question; a tool that can act
is not one. This is checked when the job is created *and* again every time it fires, so a tool that
resolves higher after an operator retunes it stops being a gate rather than
carrying on with authority nobody granted it.

### When a gate fires

| `when` | Fires when |
|--------|-----------|
| `"changed"` (default) | the whole result differs from the previous evaluation |
| `"succeeded"` | the command exits 0, or the tool call did not return an error |
| `{ matches: "<regex>" }` | the result matches the pattern |
| `{ at: "<json pointer>", is: "not-empty" \| "empty" \| "changed" }` | the pointed-at value satisfies the test |

One trap in the `"succeeded"` row: most MCP tools never set an error, so it is true on every
evaluation and the job fires every interval. It earns its place on a *shell* gate, where the exit
code is a real signal. For a tool, reach for `at` instead.

The gate's output is passed into the turn it triggers, so the model does not re-run the check the
gate just ran. A pointer narrows what is *judged*, not what the model is told: the turn still sees
the whole result, because the surrounding fields are usually what makes the fire worth reading. The
turn sees at most 8 KiB of it, and `at` will read a document up to a megabyte; past that there is
nothing for a pointer to point into and the gate reports that the probe did not return JSON. A gate
should be reading a status, not a payload.

The two shapes that compare against the previous evaluation -- `"changed"`, and `at` with
`is: "changed"` -- always fire the first time: with nothing to compare against, "changed" is the
honest answer, and it means a typo surfaces immediately instead of lying quiet. The others judge the
result on its own, so a first evaluation is no different from any other: `"succeeded"` on a command
that exits non-zero does not fire, and neither does a `matches` whose pattern is absent.

**`changed` is only as good as the stability of the result.** The check should produce something
that changes when, and only when, the watched thing does, which is a stronger requirement than
"read-only" and is where most gates go wrong. It fails in both directions. A result carrying
something that moves on its own (a timestamp, an elapsed time, a request id, an unsorted list whose
order varies) differs on every evaluation, so the gate fires every tick and costs more than the
ungated job it replaced. A result that can return to an earlier value between polls (a bare count,
where two events arrive and one is consumed) reads as unchanged, and the gate silently misses what
happened in between.

This is the reason `at` exists. Almost any JSON result carries a field that moves on its own, so
`"changed"` over the whole of it is usually wrong; `{ at: "/chats", is: "changed" }` watches the one
field you mean and ignores the `checked_at` beside it. For a shell gate, pairing a count with a
monotonic marker (`git rev-list --count HEAD` alongside the commit sha) does the same job.

> **A shell gate needs `unrestricted` permission.** It runs unattended, on a timer, until someone
> cancels it: a longer-lived grant than `execute_command`, which at least ends with the turn that
> called it. It also runs with **no sandbox**, so `workspace` cannot authorise one -- a level whose
> whole meaning is a write boundary must not hand out a command that has none. A *tool* gate is not
> held to this: `read` carries it, provided the tool resolves to `read` as well. Ungated reminders
> work at `read`.

> **`execute_command` is a tool, and that is a door.** Where a sandbox backend is usable it resolves
> to `read`, so `check: { tool: "execute_command", arguments: { command: "..." } }` is a legitimate
> tool gate at `read` -- an arbitrary command, on a timer, from a session that could not have
> authorised the shell form. What makes that acceptable is that the two are not the same thing: a
> gate dispatches at `read`, which is the level `Confinement::resolve` sandboxes, so the command
> runs read-only-confined rather than as the bare `sh -c` a `command` gate would be. Where no
> sandbox is available the same tool resolves above `read` and the gate is refused instead, so
> "admitted" and "confined" cannot come apart. The confinement blocks writes, not the network: treat
> such a gate as something that can read this machine and talk to the internet, unattended, for as
> long as the job exists.

A gate that cannot run at all -- it times out, the shell fails to start it, or its MCP server is not
connected -- is **not** treated as "nothing happened". It is logged as a warning and the occurrence
is declined, because a watcher whose check broke otherwise looks exactly like a healthy watcher with
nothing to report. The marker that tells the *agent* about it needs two consecutive failures, and
those are now an occurrence or a lease apart rather than a poll interval, so a standing breakage
takes two periods to be reported rather than twenty seconds. That is the price of not re-running a
broken check at tick cadence.

Declined means *spent*, exactly as it does for a gate that ran and said no. A recurring job moves to
its next occurrence, so a six-hour job whose server is down is probed once every six hours rather
than on every poll tick. A one-shot has no next occurrence to move to, so it keeps its claim instead
and the retry waits out `[schedule] claim_lease` (an hour by default) -- long enough that a server
restarting near the job's due time does not cost the reminder, and bounded, because each of those
retries counts against the ceiling below. Either way the gate's stored baseline is left alone, so
when the check starts working again it compares against the last value actually observed and reports
the change that happened while it was broken.

A non-zero **exit code** is different, and only `succeeded` reads it as failure: for a large class
of good gates it is the signal. `diff -q a b` and `git diff --exit-code` exit 1 exactly when there
*is* a difference; `grep ERROR log` exits 1 through the entire quiet period it is watching; `curl -f`
exits non-zero until the endpoint returns. Every other predicate judges the output and logs the exit
code at debug level, so `-vv` will show you a command that is failing when you suspect one, without
a warning on every tick of the many gates for which a non-zero exit is the normal state.

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

### Claiming an occurrence

A due job is *leased* before it runs: the host records itself and an expiry on the row, delivers the
turn, and then advances the schedule (or retires a one-shot). Three consequences worth knowing:

- **A crash costs a retry, not the job.** The row is untouched until the turn is delivered, so a
  host that dies mid-delivery leaves a lease that expires and the next host takes the occurrence.
  Before this a claim consumed the row, and a crash lost the occurrence outright -- for a one-shot,
  the whole reminder, with nothing anywhere to recover it from.
- **A cancellation always wins.** Cancelling deletes the row unconditionally; a host handing an
  occurrence back only releases its own lease, so a cancel issued while a gate is running cannot be
  undone by the handback.
- **A job that keeps failing to be delivered is parked.** Claims that end in neither a delivery nor
  a handback are counted, and after three the job stops being retried. Two things reach that count:
  a host that dies or panics mid-delivery, and a one-shot whose gate probe cannot be evaluated.
  Both leave their claim to expire rather than giving it back, so those three attempts are a lease
  apart rather than a tick apart: three panics in half a minute would otherwise park a job whose
  only problem was a blip, and nothing retires a parked recurring job afterwards. The
  job is not deleted: it stays listed, cancellable, and marked as held with the reason, because a
  prompt that crashes meka is something to look at rather than something to throw away. Recreate it,
  or cancel it. A host that simply declines a job it cannot take -- `meka serve` finding the session
  locked by a REPL -- does not count, since that says nothing about the job.

`[schedule] claim_lease` (default `"1h"`) is how long a lease is good for, and therefore how long a
crashed host's occurrence waits before another host takes it. It must exceed a gate probe plus a
turn: a lease that expires under a host still working lets a second host take the same occurrence,
which for a session-bound job is caught by the session lock but for an `isolated` job would run the
turn twice. A host refuses to start on a `claim_lease` at or under `gate_timeout`, since that half is
checkable; the turn after the probe is unbounded, so leave real headroom on top rather than treating
that check as the whole answer.

Two hosts sharing a session do not fight over its jobs. A session is held by one process at a time,
and `meka serve` leaves that session's jobs to whoever holds it rather than reaching for them and
handing the occurrence back afterwards -- which matters most for a gated job, since deciding late
would mean running its probe on every tick.

The exception is an `isolated` job, which runs in a session of its own and so needs nothing from the
one that created it. `meka serve` stays eligible for those whatever else holds the session, because
it is the only host that honours the flag. If you have a REPL open on the same session at the same
time, though, either host may take a given occurrence, and the REPL will run it in your conversation
with a warning. Run isolated jobs under `meka serve` alone if you want the flag respected every time.

### More than one host on the same database

Several meka processes pointed at one data directory -- two `meka serve` instances, or a daemon and
a terminal -- all poll the same table, so the same occurrence appears in several due lists at once.
Each occurrence is nevertheless run once. A host takes it by leasing it in a single conditional
write -- recording itself and an expiry on a row that no other host currently holds -- and the hosts
that lose that write stop before evaluating the gate: no duplicate probe, no duplicate turn, no
duplicate isolated session.
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
so holding one over is nearly free -- the sweep still evaluates whether the job is one it can run.

**What this does and does not do.** It bounds a *batch*, not a total: forty due jobs still produce
forty turns, and they are not spaced out -- a sweep that ran long leaves the next one already due.
What changes is that they arrive in groups of five, so under `meka serve` another session's single
due job is reached after five of the first session's rather than after all forty.

If you want a large backlog not to land at all, that is not what this setting is for. Cancel the
jobs (`meka schedule list`, then `meka schedule cancel <id>`) before starting a host that will fire
them, or leave `[schedule] enabled = false` while you clear it.

A **recurring** job that fires and then fails -- most often because the provider is unreachable --
leaves nothing behind in the conversation: its prompt is withdrawn, because the job produces it again
on the next occurrence. Without that, an outage would deposit one unanswered message per fire for as
long as it lasted. A **one-shot** keeps its prompt, because nothing will produce it again: its row
is retired as soon as the turn is delivered, so that message is the last trace the reminder ever
fired. A turn that got as far as running a tool keeps everything either way, since there is real work behind
it. Failures are recorded regardless: `meka serve` logs them and sends a `schedule.fired` webhook
with `status: "failed"`.

## Isolated jobs

By default a job's turn joins the conversation that created it, so you come back to a session
containing what happened while you were away.

Pass `isolated: true` and the turn runs in a fresh session instead. Isolated runs are ordinary
sessions, so `meka session list` and `meka session export` can see them.

An isolated fire runs on the **provider profile of the session that created the job**, not on the
server's default, and its fresh session records that profile like any other. A job scheduled from a
session on `work` keeps billing `work` after you point `default_provider` somewhere else, which is
the same rule the job's permission level and working directory already follow.

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

With one floor: a session at `none` fires nothing, gated or not. Nothing is executable there, so the
turn would read nothing, change nothing, and could not even reach `schedule_cancel` to stop itself
being woken again. The agent can *see* the job -- a tool's registration does not depend on the
permission level, so `[Scheduled]` still lists it and `schedule_cancel` is still offered -- but every
call is refused at dispatch, which leaves it able to describe the problem and unable to fix it. An
`every = "5s"` reminder on such a session was a turn's worth of tokens every five seconds with no
in-session way to stop it. Raise the session to restore the job; it is declined, not cancelled, and a
one-shot that came due while the session was down there is kept rather than spent.

A job's **gate** is the exception, because it runs unattended. Its bar is re-checked from two places
every time the job comes due: the level recorded on the job when it was authored, and the level the
session holds *now*. What that bar is depends on what the gate runs -- `unrestricted` for a shell
command, `read` for a tool call -- and for a tool gate the tool's own resolved level is looked up
again too, so retuning a tool's level takes effect on the next fire rather than whenever the job
is next rewritten. That level comes from `[tools.tool_permissions]` for a built-in, and for an MCP tool from the
five-step chain in [Permission resolution](../configuration/config-file.md#permission-resolution): the
server's `tool_permissions`, its `permission`, the tool's `readOnlyHint`, `[mcp] default_permission`,
then `unrestricted`. Step four is worth knowing about here: one global line turns every unannotated
tool on every server into a `read` probe a gate may call. And `readOnlyHint` is asserted by the
server and not verified by meka, which is a weaker footing under a gate than under a call in
conversation, because nobody reads the result of a gate.

That second level is the session's own, recorded on its row and kept current by whichever surface
owns it -- Shift+Tab and `/permission` in the REPL, `session/set_mode` under ACP, `PATCH
/v1/sessions/{id}` under `serve`. Every process that polls the schedule reads the same row, so
withdrawing the level works across processes: a `meka serve` daemon sharing the data directory will
refuse a gate you just dropped in a REPL. A session whose row carries no level at all falls back to
the polling process's own `--permission`. That is an ACP session that has never had
`session/set_mode` called on it, since `session/new` records no level; every other surface records
one when the session is created.

Drop the session below what the gate needs and the gate stops running -- and with it the job, because
a gate is the condition on the job and an unevaluated condition has not been met. The occurrence is
declined, and a warning is logged naming the job. Raise the session back to restore it. Unlike a gate
that ran and said "nothing happened", a held gate was never evaluated at all, so a one-shot that came
due while it was held is kept rather than spent.

The agent is told too, not just the log. A job that cannot currently fire is marked in the
`[Scheduled]` block it sees every turn and in `schedule_list`, as `NOT FIRING: <reason>`, with the
same sentence the warning carries; the moment a gate is withdrawn or restored is announced as a
world change. This matters because the two states are otherwise identical from the agent's side: a
held job and a healthy watcher with nothing to report both simply never fire. It can act on the
difference, since `schedule_cancel` needs only `read`.

A gate whose *probe* keeps breaking is marked the same way, after two consecutive failures. A server
that changed its schema, a command that was uninstalled, a pointer into a result that stopped being
JSON: each errors on every evaluation, and each is a dead watcher that looks exactly like a quiet
one. The first failure is deliberately not reported, because one failure is as often a blip as a
break. This one is tracked in memory rather than on the row, so it is known to the process running
the job: a restart re-establishes it within two poll intervals, and `meka schedule list`, which is a
separate process, does not see it.

Three surfaces report it, and each says only what it can establish:

- **`[Scheduled]` and `schedule_list`** carry the full sentence, since the agent is the one that can
  recreate or cancel the job.
- **`meka schedule list` and `/schedule`** have a `Held` column: `yes` when the job cannot fire,
  blank when it can, and `?` when this process cannot establish the answer. Blank means "it will
  fire", not "I did not check". `meka schedule list` is a separate process from any host, so it
  cannot resolve a tool gate and shows `?` for every one; `/schedule` runs inside a host and uses
  its MCP manager, so it answers them. Either shows `?` for a job whose session level it could not
  read, since that is also unestablished rather than fine. Both apply `[permissions].enabled` when
  reading a session's recorded level, so the column cannot report a job as firing that the host
  refuses.
- **`GET /v1/schedule` and `GET /v1/sessions/{id}/schedule`** carry a `withheld` field with the same
  sentence, absent when the job can fire.

Firing the reminder ungated instead would be the more forgiving-looking choice and the wrong one: it
turns a conditional job into an unconditional one, so an `every = "1m"` watcher that normally speaks
once a week would deliver a turn a minute for as long as the session stayed below that bar. An
*ungated* job is unaffected by a gate's authority, and keeps firing at any level above `none`.

At `none` nothing fires at all, gated or not. Every tool is refused at dispatch there, so the turn
would read nothing, change nothing, and be unable to reach `schedule_cancel` to stop itself being
woken again -- tokens spent to produce an agent that can describe its predicament and do nothing
about it. `POST /v1/sessions/{id}/schedule` refuses to create a job on such a session for the same
reason; `schedule_create` needs `read` to dispatch at all, so the agent cannot reach it.

Both halves are load-bearing. The recorded level rarely refuses on its own, since a creation door
already demanded it; the live level is what makes a withdrawal real. The recorded level still matters
for a job created before it was stored, which reads as "no authority" and stays refused.

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
gate_timeout = "30s"    # wall-clock budget for a gate probe
max_jobs = 50           # per-session ceiling, refused at schedule_create
max_consecutive_fires = 5 # per-session ceiling on turns spent in one sweep
claim_lease = "1h"      # how long a host's claim on an occurrence is good for
```

`max_consecutive_fires` bounds a batch, not a total. A sweep contains its turns, so lowering it does
not stop a backlog landing, nor slow it down -- it splits it into smaller groups with other sessions
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
- Keep gate probes fast and read-only. They run on every tick, and a gate that changes something
  is a side effect on a timer.
- Check what a gate's probe returns across two runs where nothing happened. Identical output is the
  whole mechanism, and anything varying inside it turns the gate into a timer.
- If a schedule matters, check what `schedule_create` reports back: it states the resolved next fire
  in absolute local time, which is how you catch a cron expression that parsed fine and means
  something other than you intended.
