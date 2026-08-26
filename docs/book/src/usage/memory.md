# Memory

Memory is the agent's own set of durable notes. It writes them itself, they survive compaction, and they outlive any single session.

Without it, an agent's only state is its context window. When a long session compacts, detail is summarised away; `conversation_search` can still search the message log, but only for something you remember to look for. Memory is the deliberate half: a fact the agent decided was worth keeping, in a place it will always see.

## How memory works

- Memories are rows in the `memories` table of meka's database (`~/.local/share/meka/meka.db`, or `MEKA_DATA_DIR`), one row per memory.
- The store is scoped to the **meka instance**, not to a session or a directory. Everything sharing a `MEKA_DATA_DIR` shares one memory; pointing a deployment at its own data dir gives it its own.
- On every prompt, meka lists each memory's `description` in the per-turn context. Bodies are **not** loaded automatically; the agent calls `memory_read` when a description suggests it needs the detail.
- The index is re-stated in full at the start of a session, after every compaction, and whenever it scrolls out of the context window. This is what makes memory survive compaction.
- Memories are available in every permission mode except **none**; all four memory tools ask only for **read**. Writing a memory therefore needs no write authority over your files, and `workspace`'s boundary does not apply to it: the store belongs to meka, not to your working tree.

> **Memories live in `MEKA_DATA_DIR`**, alongside sessions, rather than in the config directory. A backup of your config directory does not capture them; `meka memory export` is what does.

## Fields

| Field | Meaning |
|---|---|
| `name` | Unique identifier, `[A-Za-z0-9_-]`. Case-insensitive: `NOTE` and `note` are one memory. |
| `description` | One line, shown in every session's index. Make it stand on its own. |
| `priority` | `0`–`9`, default `5`. See [Priority](#priority). |
| `tags` | Lowercase labels (`[a-z0-9-]`, at most 10) for grouping and filtering. |
| `body` | Detail, loaded on demand by `memory_read`. |
| `recorded` | When the memory was made. Stamped once, at creation. |
| `updated` | When the row last changed. |
| `read count` | How many times `memory_read` has opened it. Feeds search ranking. Only `memory_read` increments it: a search hit is weaker evidence, and reading through the CLI or the HTTP API is the operator rather than the agent. |

### `recorded` versus `updated`

These answer different questions, and conflating them was a bug. A `memory_write` that changes only a description or a priority moves `updated`, and reading that as the observation date made a years-old note render as "today", sort to the top of its priority band, and arrive through `memory_read` captioned "Saved today. This is what you recorded then".

`recorded` is stamped once, when the memory is created, and carried forward untouched by every later write. It is what the index renders as an age, what ties are broken by, and what freshness weighting reads. `updated` is reported by `meka memory get` and the HTTP API and takes no part in ordering or ranking.

The rule is enforced by the `INSERT ... ON CONFLICT DO UPDATE` statement itself, which never assigns `recorded_at` on the update path, rather than by each write door remembering to preserve it.

### Omitting a field keeps what is there

`memory_write`'s `body`, `tags` and `priority` are all optional, and omitting any of them **keeps whatever the memory already had**. That makes a metadata-only update — a reworded description, say — a single call that cannot cost the note its contents, its labels or its rank. To clear the first two, pass `""` and `[]` explicitly.

`PUT /v1/memory/{name}` and `meka memory add <name> --force` follow the same rule.

## Priority

**Lower means more important** (the same direction as `nice`, the opposite of CSS `z-index`). Priority decides two things: where a memory sits in the index, and which memories survive when the index hits its size budget.

| Range | Use for |
|---|---|
| 0–1 | Standing directives that always apply |
| 2–4 | Durable facts |
| 5 | Default |
| 6–9 | Situational or short-lived notes |

Within one priority band, the most recently *recorded* memory sorts first — so a fresh note never displaces a standing rule just for being new.

Because the agent picks a priority at write time and everything feels important then, priorities tend to drift downward over a long-lived instance. `meka memory list` prints the distribution so you can see that happening and rebalance. Search ranking compensates for the same drift from the other side: see [Search](#search).

**Priority 0 is the always-in-context tier.** A priority-0 memory has its *body* rendered into the per-turn context in full, not just its description, because for a standing directive the body is the directive and leaving it behind a tool call means the agent has to look the rule up before it can follow it. The band is budgeted separately from the index (4 KiB in total, 1,024 characters per memory) so a long directive cannot crowd out the index and the index cannot crowd out the directives. Priority 1 is still "standing" for ranking purposes, but is listed by description like everything else.

Priority 0 is not a promise of unlimited space. A memory the 4 KiB band cannot fit falls through to the index below, and on a large store the index has its own ceiling to ration, so past a few dozen standing memories some of them fit nowhere. The section says so explicitly when it happens, naming how many are listed by description and how many were left out entirely, because a standing rule the agent never sees is one it is being held to and cannot read. If you see that line, either raise those notes' importance relative to the rest of the store or trim the tier: a hundred always-apply rules is not an always-apply tier.

## The index budget

The index is capped at 8 KiB and 200 entries. When more memories exist than fit, the section ends with a line stating how many were left out, and, when they carry tags, what they are about:

```
4910 more memories not shown here, most common tags infra (820), people (611),
decisions (405) — use `memory_search` to find them.
```

A bare count is not a usable signal once it runs to thousands: it says something is missing without saying what. The tag distribution is something the agent can turn into a query, which is most of what tags are for.

Nothing is lost. `memory_search` covers the whole store, including the entries the index omitted.

## Search

`memory_search` is the primary way to reach a store larger than the index can show. It is backed by a SQLite FTS5 index over the same table.

**Ranking** combines three things, so the result is what you probably meant rather than merely what matched:

- **relevance** — BM25, weighting a hit on the name above the description, and the description above the body.
- **importance** — the declared priority, blended with how often you have actually read the memory. A memory opened forty times is important whatever it was labelled two years ago, which is the counterweight to priority drift.
- **freshness** — a gentle decay on `recorded`, **disabled entirely for priority 0–1**. A two-year-old standing rule is exactly as binding as a new one; a two-year-old situational note probably is not.

**Fuzzy matching** works in four senses, and the result says which one answered so a guess is not mistaken for a recalled fact:

| Kind | Example | How |
|---|---|---|
| Word endings | `preference` finds `prefers` | Porter stemmer, always on |
| Typos and truncation | `Tokoy`, `Tok` | Retried as a prefix match, then by spelling distance |
| Word *beginnings* | `deployment` finds `deploy` | The prefix retry also works the other way |
| Unsegmented text | `深圳` inside `办公室在深圳南山区` | Retried as a literal substring |
| Different wording | `verbosity` for `terse` | Pass several phrasings in `queries` |

The second and third rows are the two the stemmer alone does not cover. SQLite's Porter strips inflections (`deploys`, `shipping`, `running`) but not every derivation: `deployment` does not stem to `deploy`, so a search for it used to miss a memory whose body says `Deploys`. The prefix retry therefore runs in both directions — shortening the *query* as well as matching the start of the stored word — and says it was a prefix match either way.

The fourth row is why word-splitting is not the whole story. The tokenizer divides on non-alphanumerics, so Chinese, Japanese and Thai prose — and a long identifier, path or URL — arrive as a single token that only matches in full. When nothing else answers, meka scans for the query as plain text instead, and says that is what it did.

The last row is the important one: `queries` is a **list**, and supplying synonyms costs nothing. `["terse", "brevity", "verbosity"]` in one call finds a memory that used any of them, which is the answer to "the agent has to guess the words it used months ago" — it does not have to guess right, only to guess several times.

Results carry enough to act on without a follow-up read: name, priority, age, read count, description, and the body itself when it is short.

### The search index

The FTS index is an [external-content](https://sqlite.org/fts5.html#external_content_tables) table over `memories`, kept in step by three triggers. It is **derived and disposable** even though the memories themselves are not:

```bash
meka memory verify              # check the index
meka memory verify --rebuild    # regenerate it from the table
```

`verify` checks two things: that the index is structurally sound, and that it holds exactly as many documents as the store does. It deliberately does not claim more. FTS5's own `integrity-check` does **not** compare an external-content index against its content table, so a memory whose text changed while a trigger was not firing leaves both checks happy — only searching for the new wording reveals it. If search is missing something you know is there, rebuild; it is one pass over the table and cannot lose a memory, because the index is derived.

## Agent tools

| Tool | Purpose |
|---|---|
| `memory_write` | Save a memory, or update one by writing to the same name |
| `memory_read` | Load one memory's body in full |
| `memory_search` | Ranked full-text search over every memory |
| `memory_delete` | Remove a memory permanently |

`memory_read` states how old the memory is and notes that it is a point-in-time observation. A memory recorded months ago is not live state, and an old note asserted as current fact is the failure this guards against. It is also the only thing that increments the read count: a search hit is weaker evidence, and an operator reading through the HTTP API is not the agent recalling anything.

`memory_write` also names an existing memory whose description says close to the same thing, when there is one:

```
Saved memory 'alice-tz' (priority 5). It is in your memory store from the next
turn on, and memory_search will find it whatever the index has room to list.

Note: 'alice-timezone' already says something very similar. If this is the same
fact, call memory_write on 'alice-timezone' instead and delete 'alice-tz' — two
near-copies both stay in the index for ever and neither supersedes the other.
```

This never blocks the write. The failure worth preventing is the silent one, where a store grows a hundred near-copies because nothing ever mentioned the ninety-nine.

## What not to save

Memory is for what is *not* derivable from the material at hand. Code structure, git history, and file contents are all reachable with `search_contents`, `read_file`, and `execute_command`, so recording them produces stale duplicates of things the agent could just look up.

What belongs in memory: who someone is and how they prefer to work, guidance you have given that should not need repeating, decisions and their reasons, and pointers to where information lives in external systems.

## CLI

```bash
meka memory list                                    # index order, plus the priority distribution
meka memory get k4yt3x-prefers-terse-replies        # every stored field
meka memory show k4yt3x-prefers-terse-replies       # the body
meka memory add tz --description "K4YT3X is in UTC+8" --priority 2 --tag people
meka memory add tz --force --description "K4YT3X is in UTC+9"   # keeps body, tags, priority
meka memory edit stale-note                         # $EDITOR on the body
meka memory remove stale-note
meka memory export --dir ~/backup/memory            # one Markdown file per memory
```

In the REPL, `/memory` lists what is saved and `/memory <name>` prints one memory's body. The listing is the table alone; the priority distribution is reserved for `meka memory list`, where you have gone looking for it.

`meka memory edit` opens the **body** only. Metadata goes through `meka memory add <name> --force --description ...`, which keeps whatever it does not mention.

## Export, backup, and git

`meka memory export` writes one `<name>.md` per memory: YAML frontmatter carrying `description`, `priority`, `recorded`, `tags` and `read_count`, followed by the body. That is the `grep`, git and backup answer now that the store is a database.

`read_count` is there because it is the one value a file cannot otherwise reconstruct. Descriptions, bodies and dates are all in the note; how often the agent has actually opened it is not, and a restored backup with every counter at zero would silently lose each memory's accumulated ranking weight.

```bash
meka memory export --dir ~/notes/memory        # must be new or empty
```

The directory must be new or empty. An export is a snapshot, and merging into an existing one would leave a stale file behind for every memory deleted since, so it would never quite match the store. An export that fails partway removes what it had written rather than leaving a truncated snapshot, which would otherwise restore as a plausible fraction of your store.

The export directory is created at mode `0700` and each file at `0600`, and an existing empty directory is tightened to `0700`. A memory body is a private note and the database it came from is `0600`; publishing the same text world-readable because that is what the umask said would be a strange way to take a backup.

What lands on disk is byte-exact: bodies, tags, priorities and recorded dates are written exactly as stored, including zero-width joiners, CRLF line endings and leading or trailing blank lines. `read_count` rides along too, because it is the one value the rest of the file cannot reconstruct.

Descriptions are the one field normalised rather than preserved: every write door collapses a description to a single line before storing it, so what comes back is what was stored. A description made only of characters YAML cannot carry has no such form, and `meka memory export` refuses the whole run and names it rather than writing a file whose frontmatter would not parse.

An export reads back with any tool that understands YAML frontmatter; meka itself has no import command, because a store you can rebuild from a directory is a second source of truth and this subsystem deliberately has one.

## Coming from a file-backed store

Memories used to be Markdown files in `<config>/memory/`. If you are upgrading from 0.41, the one-shot migration script attached to the 0.42 release imports them into the database; run it once, check `meka memory list`, then remove the directory yourself. meka never reads those files again. What it brings forward on its own is the database; a directory of files you still have is yours to import when you get to it, and importing it twice is not something a startup pass could ask you about.

## Configuration

Memory is on by default. To turn it off:

```toml
[memory]
enabled = false
```

Disabling it keeps the four `memory_*` tool schemas out of every request and renders no memory section, which is worth doing if you run lean sessions that will never use it. Memories already stored are left alone, and both the `meka memory` subcommands and the `/v1/memory` endpoints still reach them: whether an *agent* keeps memories is a different question from whether you can inspect or back up what is already there.

There is deliberately no environment variable and no CLI flag: whether an agent keeps memories is a property of the installation, not something to vary per run.
