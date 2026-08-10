# Memory

Memory is the agent's own set of durable notes. It writes them itself, they survive compaction, and they outlive any single session.

Without it, an agent's only state is its context window. When a long session compacts, detail is summarised away; `recall` can still search the message log, but only for something you remember to look for. Memory is the deliberate half: a fact the agent decided was worth keeping, in a place it will always see.

## How memory works

- Memories live in `~/.config/meka/memory/` (platform-specific config dir), one Markdown file per memory.
- The store is scoped to the **meka instance**, not to a session or a directory. Everything sharing a `MEKA_CONFIG_DIR` shares one memory; pointing a deployment at its own config dir gives it its own.
- On every prompt, meka lists each memory's `description` in the per-turn context. Bodies are **not** loaded automatically; the agent calls `memory_read` when a description suggests it needs the detail.
- The index is re-stated in full at the start of a session, after every compaction, and whenever it scrolls out of the context window. This is what makes memory survive compaction.
- Memories are available in **read**, **ask**, and **write** permission modes (not in **none**). Writing a memory does *not* require write permission: the store belongs to meka, not to your working tree.

## File format

```
~/.config/meka/memory/
├── k4yt3x-prefers-terse-replies.md
└── mekabridge-deploy-host.md
```

Each file begins with YAML frontmatter, followed by the body:

```markdown
---
description: K4YT3X prefers terse replies with no trailing summary.
priority: 2
---

Asked for this after several responses that restated the diff. Applies to
chat replies, not to commit messages or documentation.
```

| Field | Required | Meaning |
|---|---|---|
| `description` | yes | One line, shown in every session's index. Make it stand on its own. |
| `priority` | no | `0`–`9`, default `5`. See below. |

Unknown keys are ignored, so extra metadata is harmless.

## Priority

**Lower means more important** (the same direction as `nice`, the opposite of CSS `z-index`). Priority decides two things: where a memory sits in the index, and which memories survive when the index hits its size budget.

| Range | Use for |
|---|---|
| 0–1 | Standing directives that always apply |
| 2–4 | Durable facts |
| 5 | Default |
| 6–9 | Situational or short-lived notes |

Within one priority band, the most recently updated memory sorts first — so a fresh note never displaces a standing rule just for being new.

Because the agent picks a priority at write time and everything feels important then, priorities tend to drift downward over a long-lived instance. `meka memory list` prints the distribution so you can see that happening and rebalance.

## The index budget

The index is capped at 8 KB and 200 entries. When more memories exist than fit, the section ends with a line stating how many were left out:

```
18 more memories not shown here — use `memory_search` to find them.
```

Nothing is lost: `memory_search` runs a regular expression over the full text of every memory, including the ones the index omitted.

## Agent tools

| Tool | Purpose |
|---|---|
| `memory_write` | Save a memory, or replace one by writing to the same name |
| `memory_read` | Load one memory's body in full |
| `memory_search` | Regex over the full text of every memory |
| `memory_delete` | Remove a memory permanently |

`memory_read` states how old the memory is and notes that it is a point-in-time observation. A memory recorded months ago is not live state, and an old note asserted as current fact is the failure this guards against.

## What not to save

Memory is for what is *not* derivable from the material at hand. Code structure, git history, and file contents are all reachable with `search_contents`, `read_file`, and `execute_command`, so recording them produces stale duplicates of things the agent could just look up.

What belongs in memory: who someone is and how they prefer to work, guidance you have given that should not need repeating, decisions and their reasons, and pointers to where information lives in external systems.

## CLI

```bash
meka memory list                                    # index order, plus the priority distribution
meka memory get k4yt3x-prefers-terse-replies        # frontmatter and on-disk facts
meka memory show k4yt3x-prefers-terse-replies       # the body
meka memory add tz --description "K4YT3X is in UTC+8" --priority 2
meka memory remove stale-note
```

In the REPL, `/memory` lists what is saved and `/memory <name>` prints one memory's body.

Because memories are plain Markdown files in one directory, everything else works too: `grep` them, edit them in `$EDITOR`, keep the directory in git, or back it up with the rest of your config.

## Configuration

Memory is on by default. To turn it off:

```toml
[memory]
enabled = false
```

Disabling it keeps the four `memory_*` tool schemas out of every request and renders no memory section, which is worth doing if you run lean sessions that will never use it. Files already on disk are left alone.

There is deliberately no environment variable and no CLI flag: whether an agent keeps memories is a property of the installation, not something to vary per run.
