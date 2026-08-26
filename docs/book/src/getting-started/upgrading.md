# Upgrading

Most upgrades are a binary swap: replace the old executable with the new one and carry on. This page covers the ones that are not.

## 0.42 to 0.43

0.43 changes two things about the `scheduled_jobs` table, and one script does both.

It stores a scheduled job's gate differently. A gate used to be two columns, `gate_command` and `gate_fire`; it is now `gate_kind` plus a JSON `gate_spec`, which is what lets a gate call a read-only tool instead of a shell command. `migrate-0.42-to-0.43.py`, attached to the 0.43 release, converts them.

It also claims a due job by *leasing* it rather than by consuming its row, which adds `claimed_by`,
`claimed_until` and `attempts`. Those start empty, so for the lease half the migration is the columns
themselves: stop meka first and nothing is in flight to preserve. The gain is that a host which
crashes mid-delivery no longer loses the occurrence, and for a one-shot no longer loses the job.

**If you have no scheduled jobs, there is nothing to do here.** The script still adds the columns,
which every read in 0.43 names.

You cannot skip it and find out later. Every *read* of `scheduled_jobs` names `gate_kind`, so 0.43 against an unmigrated store fails with `no such column: gate_kind` and scheduling stops entirely, listings included. That is loud and harmless; the quiet failure is the one below.

Take a backup first (`sqlite3 meka.db '.backup meka.db.bak'`) and stop any running meka. The row conversion is one transaction, but Python's `sqlite3` commits `ALTER TABLE` statements outside it, so a run interrupted at exactly the wrong moment can leave the two new columns added and the rows unconverted -- which 0.43 reads as a set of *ungated* jobs. Re-running the script fixes that; the backup is for the interruption that does not get a re-run.

Coming from 0.41, run `migrate-0.41-to-0.42.py` first. This script checks and refuses a 0.41 store rather than converting it into a shape 0.43 still cannot read.

```bash
python3 migrate-0.42-to-0.43.py            # reports what it would change; writes nothing
python3 migrate-0.42-to-0.43.py --apply    # does it
```

Order is looser than the 0.41 upgrade: this script adds the two columns itself, so it can run before or after you install 0.43. What it must not do is run *after* 0.43's scheduler has been let loose on the store.

### Read the `!` lines

A row 0.42 refused to load -- a half-written gate, or a `gate_fire` value it did not recognise -- cannot be converted, so the script reports it and leaves it alone. That row is the quiet failure: 0.43 reads a NULL `gate_kind` as "no gate at all", so an `every = "30s"` watcher that used to fire on a condition becomes one that fires **every thirty seconds**. Cancel those jobs with `meka schedule cancel <id>`, or recreate them with a gate, before starting 0.43's scheduler. The script prints the count and the ids.

Everything else is preserved, including each gate's stored baseline, so a `changed` gate does not fire spuriously on its first evaluation after the upgrade. Re-running the script is a no-op.

## 0.41 to 0.42

A store written by 0.41 needs five conversions before 0.42 reads all of it. They are performed by `migrate-0.41-to-0.42.py`, a one-shot script attached as an asset to the 0.42 release. Download it, run it once, and you are done with it.

meka does not convert anything at startup, and that is deliberate: a migration that runs on every start is one nobody can see fail. A script runs under your eye, tells you what it would change before it changes anything, and finishes.

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
