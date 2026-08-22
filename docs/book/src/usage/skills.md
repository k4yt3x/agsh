# Skills

Skills are knowledge packages that give the agent non-standard knowledge: manuals, procedures, tool-specific instructions, and experience the LLM doesn't have natively. Each skill is a directory containing a `SKILL.md` file with structured metadata.

meka implements the [Agent Skills specification](https://agentskills.io/specification), so a skill written for meka works in other compliant clients and vice versa, and a skill meka writes passes the ecosystem's own `skills-ref validate`.

Skills are normally authored by you. An agent can also be allowed to write its own; see [Letting the Agent Manage Skills](#letting-the-agent-manage-skills), which is off by default.

## How Skills Work

- Skills live in `~/.config/meka/skills/` (platform-specific config dir). Additional read-only directories can be added with [`extra_paths`](#reading-skills-from-other-directories).
- Each skill is a directory: `skills/<name>/SKILL.md`. A lowercase `skill.md` is accepted too.
- Any entry whose name begins with `.` is skipped at discovery. This covers VCS metadata (`.git`), editor/IDE state (`.vscode`, `.idea`), filesystem artifacts (`.DS_Store`, `.Trash`), and any other dotfile or dotdir that may sit alongside your skills.
- `SKILL.md` starts with a YAML frontmatter block declaring the skill's metadata, followed by Markdown body content.
- On every prompt, meka discovers all valid skills and lists them in the per-turn context with their `description`.
- The agent invokes a skill by calling the `skill_read` tool with the skill name. The tool returns the full body, which the agent follows.
- `skill_search` greps the full text of every installed skill, for when the one-line descriptions are not enough to tell which skill covers something.
- Skills are available in **read**, **ask**, and **write** permission modes (not in **none**).
- The whole subsystem can be switched off with `[skills] enabled = false`, which keeps the skill tools' schemas out of every request and stops the skills section from rendering.

## File Format

A skill is a directory under `~/.config/meka/skills/` containing a `SKILL.md` file:

```
~/.config/meka/skills/
└── download-videos/
    └── SKILL.md
```

`SKILL.md` must begin with a YAML frontmatter block, followed by the skill body:

```markdown
---
name: download-videos
description: Download videos from various websites using yt-dlp. Use when the user wants a video off a URL.
metadata:
  author: John Doe <john.doe@example.com>
  version: "1.0"
---

# Download Videos with yt-dlp

## Installation

Install yt-dlp:

\```bash
pip install yt-dlp
\```

## Basic Usage

Download a video:

\```bash
yt-dlp "https://example.com/video"
\```
```

### Required Frontmatter Fields

| Field | Constraints |
|-------|-------------|
| `name` | 1-64 characters, lowercase alphanumerics and hyphens; no leading, trailing or consecutive hyphens. Must match the directory name. "Alphanumeric" is Unicode-wide, as the spec and its reference validator define it, so a non-Latin name is valid. |
| `description` | 1-1024 characters. What the skill does *and when to invoke it*. Shown to the model in the per-turn context, so fold the trigger condition into this one line. |

A skill is **skipped, and the reason reported**, when it breaks a rule the spec states about its identity: a directory name outside the rules above, a `name` that disagrees with its directory, or a missing `description`. meka implements the Agent Skills specification, so a directory it cannot read as a conforming skill is not a skill it has, and saying so beats listing something no other client would accept.

A skip is not silent. It appears in the [`[Skills]` index](#how-the-agent-uses-skills) the agent reads, in `skill_read`, in `meka skill get`, and in a warning on startup, each naming the directory and the reason. The directory stays where it is, so renaming it is all that is needed.

Everything else loads: an over-long description (warned, since refusing would take the procedure with it), and frontmatter keys the spec does not define.

A name containing characters meka cannot render -- a newline, a zero-width space -- is refused by the same rule, and additionally cannot be reached by `meka skill remove`: the name meka would echo back is a *different* directory, which may itself exist. Rename it in a shell.

Keys meka does not model are **kept, not dropped**. A skill carrying Claude Code's `when_to_use`, or a `source_url` from a meka older than this one, still has them after an agent edits its description.

Frontmatter that is not valid YAML is **not** repaired. The client guide suggests quoting unquoted prose colons (`description: Extract text. Use when: the user mentions PDFs`) as a fallback, and meka deliberately does not: the reference implementation does no repair either, one repair rule is an arbitrary pick out of the many ways YAML can be malformed, and a file meka silently fixed on the way in is one that keeps working here and nowhere else. The skill is skipped instead, and [said so out loud](#how-the-agent-uses-skills) with the parser's own line and column. Fix the file once and every client can read it.

### Optional Frontmatter Fields

| Field | Description |
|-------|-------------|
| `license` | The skill's license, as a name or a reference to a bundled file. Informational. |
| `compatibility` | Up to 500 characters naming what the skill needs from its environment (`Requires Python 3.14+ and uv`). Shown to the model when the skill is activated, since it changes how the instructions should be carried out. |
| `allowed-tools` | Tools the skill would like pre-approved. **meka reads and preserves this but never acts on it**; see [Why `allowed-tools` Is Ignored](#why-allowed-tools-is-ignored). Written as a space-separated string; a YAML list or a bare number is read too, rather than costing you the skill. |
| `metadata` | A map of extra properties. Where anything the spec has no field for belongs. |

### Metadata Keys

The spec reserves `metadata` for properties it does not define, and meka carries the whole map through untouched, including keys it has no meaning for, so a skill written elsewhere survives being edited here.

That includes **values that are not strings**. The spec describes a map of string to string, but skills in the wild carry lists and nested maps under `metadata`, and an edit here keeps them as they were:

```yaml
metadata:
  tags: [pdf, forms]     # still a list after an agent rewrites the description
  origin:
    repo: example/skills # still a map
```

meka renders such a value as text where it needs one (`meka skill list`, `meka skill get`), but the file keeps the original.

`metadata` itself must be a map, though. A skill whose `metadata:` is a string or a list still loads, lists and reads normally, but **rewriting it is refused**: meka would have nowhere spec-legal to record `meka-priority` or `author`, and doing something other than what the caller asked without saying so is worse than declining. Fix the file, or write to a different name.

| Key | Default | Description |
|-----|---------|-------------|
| `author` | none | Attribution, conventionally `Name <email>`. The spec's own example key. Informational only. |
| `version` | none | Free-form version label (e.g. `"1.0"`, `"2024-03-14"`). The spec's own example key. |
| `meka-priority` | `5` | Listing rank `0`-`9`, lower first. Orders the `[Skills]` index and decides which skills its cap drops. Not shown to the model; see [How the Agent Uses Skills](#how-the-agent-uses-skills). |

`meka-priority` carries a prefix because it is meka's own concept and the spec has nothing like it; another client could reasonably read a bare `priority` the opposite way round. `author` and `version` do not, because the spec demonstrates exactly those keys.

### Why `allowed-tools` Is Ignored

`allowed-tools` is experimental in the spec, and the spec itself notes that support varies. meka parses it, preserves it across a rewrite, and shows it in `meka skill get`, but never grants anything from it. The spec defines the field as a space-separated string and that is what meka writes, so a skill that spelled it as a YAML list comes back joined: `[Read, Bash]` becomes `Read Bash`.

Two things meka does not preserve across a rewrite, both worth knowing before you hand-edit a `SKILL.md` that meka will later write to:

- **A joined `allowed-tools` entry containing a space cannot be told apart from two entries.** `["Bash(git diff:*)", "Read"]` comes back as `Bash(git diff:*) Read`, which no longer says where one entry ends. Prefer the spec's string form for these.
- **Comments in the frontmatter block are dropped.** The header is rebuilt through a YAML serializer, which does not carry them. Keys, values and nested structure all survive; `# notes` beside them do not. Put anything you need to keep in the body, which is passed through untouched.

A skill file is content, and content does not get to widen what the agent may run. meka's [permission mode](./permissions.md) is the authority for that, and a `SKILL.md` dropped into the skills directory (or synced from a repository, or written by an agent) must not be able to pre-approve `Bash(rm:*)` on its own say-so.

### Referencing Bundled Files

Refer to files bundled alongside `SKILL.md` by relative path (e.g. `scripts/helper.sh`). Every skill body is prefixed with a header naming the skill's directory (see [How the Agent Uses Skills](#how-the-agent-uses-skills)), so relative paths resolve against the skill rather than against the session's working directory.

The body is passed to the model verbatim; meka does not rewrite anything inside it. Keeping skills free of host-specific placeholders is what lets the same `SKILL.md` run under meka and other Agent Skills hosts unchanged.

## Storage Location

| Platform | Path |
|----------|------|
| Linux | `~/.config/meka/skills/<name>/SKILL.md` (`$XDG_CONFIG_HOME/meka/skills/`) |
| macOS | `~/Library/Application Support/meka/skills/<name>/SKILL.md` |
| Windows | `%APPDATA%\meka\skills\<name>\SKILL.md` |

This is the only directory meka ever *writes* to. It is created the first time something is written there, not at startup.

## Reading Skills from Other Directories

`[skills] extra_paths` adds directories to scan. They are **read-only**: meka never creates them and never writes into them, so listing one costs nothing if it does not exist.

```toml
[skills]
extra_paths = ["~/.agents/skills", "~/src/team-skills/skills"]
```

`~` is expanded. A relative path resolves against the process working directory.

The default is empty. `~/.agents/skills` has emerged as a cross-client convention, so pointing at it makes skills installed by other Agent Skills clients visible to meka, but whether to read a directory outside meka's own namespace is your decision rather than a default.

**Precedence.** meka's own store is searched first, then each `extra_paths` entry in order. When two directories hold the same skill name, the first wins and the shadowed one is logged.

**Writes never follow.** `skill_write`, `skill_delete`, `meka skill add`, `meka skill remove`, `PUT /v1/skills/{name}` and `DELETE /v1/skills/{name}` all target meka's own store. Asked to write a name that resolves to a skill in a read-only root, they refuse and say where it lives, because writing would create a second copy that shadows the original instead of changing it. Edit that file directly, or pick a different name. This holds whether or not the file there is valid: a directory whose `SKILL.md` does not parse still claims that name, and shadowing a broken skill is the case worth refusing hardest, since nothing then reports the original at all.

There is deliberately no automatic project-level scan. meka does not treat the working directory as trusted anywhere else either, and a cloned repository that could silently add instructions to the agent's context would be exactly that. Name a project's skills directory in `extra_paths` if you want it read.

## Listing Skills

`meka skill list` shows a fixed set of columns, so output stays parseable when piped:

```console
$ meka skill list
Name             Author                Pri  External  Description
deploy-service   Jane Doe              2    false     How to deploy the service. Use when the…
borrowed         -                     5    true      A skill another client installed.
```

`External` is `true` for a skill found under [`extra_paths`](#reading-skills-from-other-directories) rather than in meka's own store. It is always present, even when nothing is external, so a script's field offsets do not shift with the store's contents.

`--paths` adds the on-disk `Path`, which is how you find out *where* an external skill lives. Nothing else goes in this table: `license`, `compatibility`, `allowed-tools`, `version` and arbitrary `metadata` keys are per-skill detail, and `meka skill get <name>` prints all of them.

## How the Agent Uses Skills

When skills are available, the per-turn context includes a `[Skills]` section like:

```
[Skills]

- **download-videos**: Download videos from various websites using yt-dlp. Use when the user wants a video off a URL.
- **deploy-kubernetes**: Deploy services to a K8s cluster. Use when the user asks to deploy to Kubernetes.
```

The list is sent once, not on every turn. Adding, editing, or removing a skill mid-session is picked up on the next prompt and announced as a short note naming just what changed, so a long session doesn't pay for the whole list repeatedly.

A skill directory that could not be loaded is named there too, with the reason:

```
1 directory in your skills path could not be loaded, so it is not in the index above and cannot be invoked:

- **deploy-kubernetes**: invalid frontmatter: mapping values are not allowed here
```

This is the counterpart to the same paragraph in `[Memory]`, and it exists because the log is not a channel the agent can read. From inside a session an unparseable `SKILL.md` is otherwise indistinguishable from a skill nobody wrote: the index omits it, so the agent has no reason to ask for it by name, and whoever dropped the file in goes on believing the procedure is in force. Naming it lets the agent tell you rather than improvise a replacement. It appears and disappears with the file, so repairing the frontmatter is announced too.

Skills are listed in `meka-priority` order, lowest first, with the name breaking ties. The index is capped at 200 entries and 8 KiB; anything past that is replaced by a count and a pointer to `skill_search`, so a large skill store degrades into "search me" rather than silently eating the context window. The rank itself is not rendered: a skill should be invoked because the request matches its stated purpose, not because it outranks another one.

The agent loads a skill by calling the `skill_read` tool:

```
skill_read(name: "download-videos")
```

The tool returns the full body of `SKILL.md` as its output. The agent then follows the instructions.

Whenever a skill body is loaded (by the `skill_read` tool, `--skill`, `/skill`, `agent_spawn`, or `meka skill show`), it is prefixed with a header naming the skill's directory:

```
Base directory for this skill and its bundled files: /home/user/.config/meka/skills/download-videos
```

This is what lets the agent locate files bundled alongside `SKILL.md` when the body refers to them by relative path (e.g. `scripts/helper.sh`).

A skill that declares `compatibility` gets a second line, since what the skill needs from its environment changes how its instructions should be carried out:

```
Environment this skill expects: Requires Python 3.14+ and uv
```

## Running a Skill in a Sub-Agent

The agent can delegate a skill to a sub-agent by passing the `skill` parameter to the `agent_spawn` tool. The sub-agent runs the skill in its own fresh context and returns a report, keeping the skill's instructions out of the parent's conversation:

```
agent_spawn(skill: "summarize-financial-news")
agent_spawn(skill: "summarize-financial-news", prompt: "focus on UK markets")
```

`prompt` is optional when `skill` is given; if both are supplied, `prompt` is prepended to the skill body as extra direction (the same ordering as `meka --skill <name> [prompt]`).

A skill is the reusable unit of worker instruction. Sub-agents do not receive the [instructions file](./instructions.md) unless the spawn call asks for it, since it describes the top-level agent rather than a delegate — so a skill is usually the better way to give a worker standing direction.

## Invoking a Skill from the CLI

Any skill can be triggered directly from the command line with `--skill <name>`. The rendered body becomes the first user turn, and meka drops into the interactive REPL after the turn finishes:

```bash
meka --skill download-videos "https://example.com/video"
```

The positional `[PROMPT]` argument, if given, is prepended to the skill body as extra context (equivalent to typing `/skill download-videos https://example.com/video` in the REPL).

To run the skill and exit immediately (useful for scripts), pair with `--oneshot`:

```bash
meka --oneshot --skill download-videos "https://example.com/video"
```

To invoke a skill mid-session inside the REPL, use the slash command instead:

```
/skill download-videos
/skill download-videos this URL specifically
```

## Letting the Agent Manage Skills

By default the agent can only read skills. Setting `[skills] agent_managed = true` additionally
registers `skill_write` and `skill_delete`, letting it create, refine, and remove skills itself:

```toml
[skills]
agent_managed = true
```

This is off by default because for an ordinary terminal session you author and curate skills, and an
agent rewriting that store is not something you asked for. It earns its keep in the opposite
deployment: a long-running agent acting as a dispatcher over a team of sub-agents. A skill is the
only thing in meka that both outlives the session and can be handed to a sub-agent as its task, so
writing one is how such an agent gets a refined worker brief to the next worker without routing the
whole text through its own context window.

```
skill_write(name: "triage-build-failure",
            description: "How to triage a failing CI build",
            priority: 2,
            body: "1. Fetch the log...")
agent_spawn(skill: "triage-build-failure")
```

Notes on how it behaves:

- Both tools run at **read** permission, like `memory_write`. They write to meka's own config
  directory rather than to your working tree, and the deployment they exist for typically runs at
  read permission permanently. The config flag is the authorization, not the permission tier.
- **Sub-agents never get them**, whatever this setting says. A worker that inferred something from
  one narrow task should not rewrite the instructions its siblings run on.
- Writing to an existing name updates it. Omitting `body` keeps whatever the skill already
  documented, so a call that only changes the description or priority does not erase the procedure.
  Note that `meka skill add --force` is a *replace*, not an update: it rewrites `SKILL.md` from the
  template and removes any bundled files alongside it. The new `SKILL.md` is written first, so a
  failure to clear a bundled file leaves the skill intact and says which files it could not remove.
- Skills the agent *creates* are stamped `metadata.author: meka (agent-authored)`, so
  `meka skill list` shows where an entry came from. An existing `author` is kept, so an agent
  refining a skill you wrote does not reassign it to itself. Informational, not a guard.
- **Every other frontmatter key survives a rewrite**, including `license`, `compatibility` and any
  `metadata` key meka has no meaning for, with its YAML type intact, so a `metadata` list stays a
  list. An agent asked to sharpen an imported skill's description changes the description and
  nothing else.
- The confirmation reports the rank the *file* ended up with, not the one the call asked for. The
  two differ only when the skill's `metadata:` is not a map, which meka will not overwrite; the tool
  says so rather than claiming a change that did not happen.
- A file that exists at that name but is not a valid skill is **refused, not overwritten**. Such a
  file is invisible everywhere else in meka, so nothing could tell you what was about to be lost.
- A skill from a read-only [`extra_paths`](#reading-skills-from-other-directories) root is
  **refused** by both tools, since writing would shadow it rather than change it.
- A hand-written skill in meka's own store is not protected from being rewritten. The flag being off
  by default, and your config directory being in version control, is the safety net for that case.
- `skill_delete` removes the whole skill directory, including any bundled files, matching
  `meka skill remove`.

## Tips

- Use short, unambiguous skill names (e.g. `setup-postgres`, not `pg`). The name is what the agent sees and calls, and the spec allows only lowercase alphanumerics and hyphens.
- **Anything meka lists, meka can remove**, and so is almost anything it refuses to load. A name the spec forbids (`My_Skill`, `two words`, `not.a.skill`) is skipped with the reason named, and `meka skill remove` still takes it so you can clean up. The one exception is a name meka cannot [render](#required-frontmatter-fields), which no command can address; rename it in a shell. One Windows reserves, like `con`, loads normally: that is meka's own write-time rule, not the spec's.
- Every write door applies the same rules. `meka skill add`, `skill_write` and `PUT /v1/skills/{name}` all refuse a name or a description the spec rejects, and refuse a skill whose `name` is missing or disagrees with its directory, so a skill meka authors passes `skills-ref validate`. `--from-file` copies your bytes verbatim, so it can still carry a key the spec does not define -- that is how an imported skill keeps its `when_to_use` -- but it must still declare the required `name`. Run `uvx skills-ref validate <dir>` when you want the reference's own verdict on a file.
- Write `description` concisely, and fold the "use when..." trigger into it. It is sent to the model and consumes tokens.
- Keep each skill focused on a single topic or procedure. Spawn multiple skills rather than one giant one.
- Bundle supporting files in the skill directory and reference them by relative path (`scripts/file.ext`).
- Skills are re-discovered on every prompt, so you can add, edit, or remove skills mid-session without restarting meka.
