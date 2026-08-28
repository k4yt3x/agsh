# Upgrading

Most upgrades are a binary swap: replace the old executable with the new one and carry on. This page covers the ones that are not.

## 0.43 to 0.44

A binary swap, and the store migrates itself as promised below, **unless you authenticate an MCP
server with `auth_token` or `client_secret`**, which are no longer config keys. Read the next
section first if you do; meka will refuse to start otherwise. Then two behaviour changes worth
knowing before you resume an existing session, and two more if you run `meka serve` or `meka acp`.

**MCP secrets moved out of `config.toml`.** `auth_token` on a server, and `client_secret` in a
`[mcp.servers.auth]` block, are gone. Both were secrets sitting in a plaintext file people commit
and sync; they now live in meka's database beside the OAuth tokens, which is where provider
credentials have always been.

meka cannot move them for you. The store migrates itself because it has a ledger recording what it
has already done; `config.toml` has none and may be older or newer than the binary at any moment, so
a key left behind is a parse error naming the key and the line rather than a value silently ignored:

```console
$ meka mcp list
Error: configuration error: failed to parse …/config.toml: TOML parse error at line 12, column 1
   |
12 | auth_token = "…"
   | ^^^^^^^^^^
unknown field `auth_token`, expected one of `name`, `transport`, …
```

For each server, delete the line and store the secret instead. Which command depends on which key
you deleted, and the two are alternatives, not a sequence: a bearer belongs to a server with no
`[auth]` block, a client secret to one that has it.

```console
$ # for a server whose `auth_token` you deleted (no [auth] block):
$ pass show api-token | meka mcp login api --auth-token-stdin

$ # for a server whose [auth] block's `client_secret` you deleted:
$ pass show acme-secret | meka mcp login acme --client-secret-stdin
```

`meka mcp get <name>` then lists the kinds it holds without printing any of them. `--auth-token` and
`--client-secret` are gone from `meka mcp add` for the same reason: an argument is visible in `ps`
output and in the shell history of every user on the machine. Use the `-stdin` forms, which `add`
also takes.

If you were using `auth_token = "${API_TOKEN}"` to keep the token out of the file, a header does the
same job and still expands: `headers = { Authorization = "Bearer ${API_TOKEN}" }`. Storing it is the
better answer, since it survives without the variable being set.

Nothing else about a server moves. `env`, `args` and `headers` stay in `config.toml` with `${VAR}`
expansion, because they configure a process or a request and merely *may* contain a secret.

**`isolated` scheduled jobs are gone; every job fires in the session that created it.** The mode ran
a job's turn in a fresh session rather than the conversation that made it, to avoid replaying that
conversation's history. Only `meka serve` ever honoured it: the REPL and ACP already ran such a job
in the open conversation, with a warning, so for two of the three hosts nothing changes at all.

Existing jobs are not deleted and do not need touching. The store drops the column and the job keeps
its schedule and its prompt, firing into the session it belongs to from then on.

What it cost is why it went. The fire inherited the creating session's authority (its permission
level, its working directory, its provider profile, its MCP servers) and dropped the conversation,
which is where anything you told the agent that never reached a memory or an instructions file
lives. Its result landed in a session nothing linked to, and the turn could not even cancel its own
job, because `schedule_cancel` resolves against the session it is running in.

`meka acp` and `meka serve` clients: `POST /v1/sessions/{id}/schedule` now rejects `isolated` with a
422 naming the field, rather than accepting and ignoring it. `GET /v1/schedule` and the
`schedule.fired` webhook no longer carry it either.

If you were relying on the mode, an external timer does the same job with the level and profile
stated outright instead of inherited:

```bash
meka --oneshot --permission read --provider work "summarise today's alerts"
```

Often a gate is the better answer: it means a frequent job takes no turn at all on the ticks where
nothing happened, which saves more than skipping the history did.

**A scheduled job is refused on a sub-agent session.** `POST /v1/sessions/{id}/schedule` answers 422
if the session was spawned by another. Sub-agents never had the `schedule_*` tools, so no job meka
created can be affected; what this closes is a client planting one directly, which would have woken
the worker without the tool restrictions or memory grants it was spawned under.

**A resume now starts at the level the session recorded.** Both CLI hosts do this: the REPL and
`meka --oneshot -c` / `-r`. The scripted one is where a silent change matters most, since a
`--oneshot` run that passes no `--permission` used to start at the config default and now starts at
whatever the session was last set to. A session you created with `--permission unrestricted` comes
back at `unrestricted` without the flag. Before, the row said one thing and the run did another;
every other surface already read the row, and these two were the ones that did not. Pass
`--permission` on the resume to move it. A level that is no longer in `[permissions].enabled` is not
granted: the session drops to the configured default with a warning.

**A session now runs on the provider profile it was created with.** Every existing session is
recorded as running on your current default profile, which is what they were in fact running on, so
nothing moves. From here `meka -p openai` then `meka -c` stays on `openai`. If nothing could be
resolved when the migration ran (no profile configured yet), sessions are left without one and say
so; resume such a session once with `--provider <name>` to record it. The migration says which
profile it recorded and on how many sessions; run once with `-v` if you want to see it.

**`GET /v1/info` no longer returns `provider` or `model`.** Read them from `GET /v1/providers`
instead, which lists every configured profile with its `name`, its `type` (the backend), its
`model`, and `active: true` on the one a session gets when it names none. The old fields held the
default profile's *backend* under the name `provider`, while `provider` on `POST /v1/sessions` names
a *profile*, so a client that read one and posted it to the other got a 422. They were duplicates of
the `active` row besides.

**`meka acp` and `meka serve` refuse `-c`, `-r`, `--model` and `--base-url`.** All four name one
run's session, and a long-lived host has no such thing: it creates one per `session/new` or per
`POST /v1/sessions`, each naming its own profile. They used to be accepted and quietly misapplied,
which was worse than it sounds: `GET /v1/info` went on reporting a `--model` no session ever used,
and `-c` / `-r` switched off the default-profile check a host with no default needs most. Set
`model` and `base_url` on the profile in `config.toml`, or name a `provider` per session. `--provider`
is still accepted, because it selects which configured profile the host defaults to, which is a
property of the host rather than of one session.

Also on the HTTP side, and not a break: `PATCH /v1/sessions/{id}` with a body naming only a provider
now works on a session that is not loaded, which is how you move one whose profile has left
`config.toml`. It takes the session lock to do it, so if you run more than one `meka` on the same
store, send it to whichever process has the session; another one answers `409` `session-locked`
rather than moving a row the running host would ignore.

## 0.42 to 0.43

Nothing to do. Start 0.43 and it brings the store forward itself, on the first open, before anything reads it.

This is the first release that migrates its own store, and from here on that is the rule: upgrades from 0.43 onward are a binary swap, whatever the schema does.

What it changes, if you want to know what happened. A scheduled job's gate used to be two columns, `gate_command` and `gate_fire`; it is now `gate_kind` plus a JSON `gate_spec`, which is what lets a gate call a read-only tool instead of a shell command. And a due job is now claimed by *leasing* it rather than by consuming its row, which adds `claimed_by`, `claimed_until` and `attempts`, so a host that crashes mid-delivery no longer loses the occurrence, or for a one-shot the whole job. Each gate's stored baseline is preserved, so a `changed` gate does not fire spuriously on its first evaluation afterwards.

Before it writes anything, meka copies the store to `meka.db.v1.bak` beside it. That doubles the space the store takes until you delete it, which is worth knowing if yours is large. Start with `-v` once if you want the exact path in the log; the copy is otherwise silent, and nothing prunes it. It records the version it was taken at, so if you ever restore it, the next start migrates it again correctly rather than mistaking it for a store that is already current.

The whole thing is one transaction, so an interruption leaves the store exactly as it was rather than half-converted. Running two hosts at once is fine: the first takes the schema lock and the second waits, then finds nothing to do.

**Coming from 0.41 or older**, run `migrate-0.41-to-0.42.py` once first, as described below. 0.43 recognises a 0.41-shaped store and refuses it by name rather than converting it into something still unreadable, and it changes nothing when it does.

### A gate that cannot be read

Rare, and worth knowing the shape of. If a job's gate was already unreadable under 0.42 (a hand-edited row, or a `gate_fire` value meka never wrote), it cannot be converted, because there is nothing to convert it *from*. Such a job never fired under 0.42, and it does not fire under 0.43 either: the migration leaves it in the same refused state rather than guessing at what it meant or deleting it. It is logged once, by id, at `warn`.

The consequence is that the row stays inert and invisible, as it already was: it will not appear in `meka schedule list` and `meka schedule cancel` cannot reach it. Recreate the job if you still want it. The original row is in the backup.

## 0.41 to 0.42

A store written by 0.41 needs five conversions before 0.42 reads all of it. They are performed by `migrate-0.41-to-0.42.py`, a one-shot script attached as an asset to the 0.42 release. Download it, run it once, and you are done with it.

This one stays a script, and 0.43's own store migration does not replace it: 0.42 carried no migration code to reach back with, and conversion B below has to *guess*. 0.41 recorded nothing about which provider a thinking block came from, so the script tells them apart by the shape of the blob, and it reports what it read before it writes. A guess wants a human reading the counts, which is the one thing a migration that runs on every start cannot offer.

### Order

1. **Run 0.41 once, before you replace it.** It brings a store from an older release fully up to date; 0.42 carries no migration code and cannot.
2. **Install 0.42 and launch it once.** This is what creates the tables the script writes into, so it is not an arbitrary step you can move: run the script against a database that predates 0.42 and it stops with an explanation rather than guessing.
3. **Run the script**, first as a dry run, then with `--apply`.

```bash
python3 migrate-0.41-to-0.42.py            # reports what it would change; writes nothing
python3 migrate-0.41-to-0.42.py --apply    # does it
```

Read the dry run before you apply it. Conversion B in particular reports how many thinking blocks it read as Claude's and how many as OpenAI's, and 0.41 did not record which was which. If those counts do not match the providers you actually used, stop: the blocks it could not place are left alone, but the ones it places wrongly are not recoverable from the row afterwards.

The dry run is the only place to read that. Its per-class counts and its warning about a session holding both kinds describe the write it is about to do, so once the blocks are converted a later run has nothing left to report about them.

Between steps 2 and 3 the store is live but incomplete: memories are absent from the agent's index, and any session affected by conversion E below is already broken. Step 3 is part of the upgrade rather than cleanup to get to later.

The script finds meka's own directories by default, honouring `MEKA_CONFIG_DIR` and `MEKA_DATA_DIR`; `--root`, `--skills-root` and `--database` point it at a copy instead. `--self-test` checks the script against its own fixtures and exits, touching nothing of yours.

### What it converts

| Conversion | What it changes | If you skip it |
|---|---|---|
| **A.** Memories | The Markdown files under `<config>/memory/` become rows in the database's `memories` table, which is where 0.42 reads memories from. The files are read, never written or deleted. | The memories are simply not there. The files are untouched on disk, so nothing is lost and the import still works whenever you get to it. |
| **B.** Thinking blocks | A stored block's bare `signature` becomes an `opaque` object naming which provider it belongs to: `signed` for a Claude signature, `sealed` for OpenAI's encrypted reasoning. 0.41 wrote both to the same field and recorded nothing about which was which, so the script tells them apart by the shape of the blob and **reports the counts before it writes**. A blob it does not recognise is left exactly as it is. | The block loses its opaque half, so that reasoning stops being replayed to the provider. The session still loads and still runs; it just resumes without the chain of thought behind those turns. |
| **C.** A skill's `version:` / `author:` | Both move from the top level of a `SKILL.md`'s frontmatter under `metadata:`, keeping their names, which is where the Agent Skills spec puts them. | Nothing. meka reads a top-level `version:` and `author:` permanently, because Claude Code's plugin skills declare `version:` there. This conversion is cosmetic. |
| **D.** A skill's `priority:` | Moves under `metadata:` **and is renamed** to `meka-priority:`. | The skill silently drops to the default rank of 5. A rank is read from `metadata.meka-priority` and nowhere else, so the `[Skills]` index comes out in a different order and its cap drops different skills. Nothing warns. |
| **E.** A stored `tool_result` | Content held as a bare JSON string becomes a list of typed blocks, `[{"type": "text", "text": ...}]`. | The affected session breaks. The row will not deserialize, so it is dropped as the session loads, which orphans the `tool_use` it answered, and the provider then refuses the next turn. |

### The two that matter

A and B announce themselves: a memory you saved is missing from the index, or a thinking block is not replayed. Both are recoverable by running the script later.

**D and E are the ones that damage silently.** D changes which skills the `[Skills]` index shows first and which its cap drops, with nothing on screen to say the rank it used was not the one in your file. **E can leave a session unusable**: it loads cleanly, and then the next turn is refused by the provider because a `tool_use` in the history has no matching result. See [Sessions](../usage/sessions.md#rewinding-a-session) if you have already met that error.
