# Skills

Skills are knowledge packages that give the agent non-standard knowledge: manuals, procedures, tool-specific instructions, and experience the LLM doesn't have natively. Each skill is a directory containing a `SKILL.md` file with structured metadata.

Skills are normally authored by you. An agent can also be allowed to write its own; see [Letting the Agent Manage Skills](#letting-the-agent-manage-skills), which is off by default.

## How Skills Work

- Skills live in `~/.config/meka/skills/` (platform-specific config dir).
- Each skill is a directory: `skills/<name>/SKILL.md`.
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
description: Download videos from various websites using yt-dlp. Use when the user wants a video off a URL.
version: "1.0"
author: John Doe <john.doe@example.com>
source_url: https://gist.githubusercontent.com/k4yt3x/.../raw/SKILL.md
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

| Field | Description |
|-------|-------------|
| `description` | Summary of what the skill does *and when to invoke it*. Shown to the model in the per-turn context. Fold the trigger condition into this one line. |

Skills missing `description` are skipped at discovery with a warning log. Unknown frontmatter keys are ignored, so a skill authored for Claude Code (which carries extra keys like `when_to_use` or `allowed-tools`) still loads.

### Optional Frontmatter Fields

| Field | Default | Description |
|-------|---------|-------------|
| `version` | none | Free-form version label (e.g. `"1.0"`, `"2024-03-14"`). |
| `author` | none | Attribution, conventionally `Name <email>` (e.g. `John Doe <john.doe@example.com>`). Informational only. |
| `priority` | `5` | Listing rank `0`-`9`, lower first. Orders the `[Skills]` index and decides which skills its cap drops. Not shown to the model; see [How the Agent Uses Skills](#how-the-agent-uses-skills). |
| `source_url` | none | An `https://` URL the skill's `SKILL.md` can be re-fetched from. Enables [`meka skill update`](#updating-skills). |

### Referencing Bundled Files

Refer to files bundled alongside `SKILL.md` by relative path (e.g. `scripts/helper.sh`). Every skill body is prefixed with a header naming the skill's directory (see [How the Agent Uses Skills](#how-the-agent-uses-skills)), so relative paths resolve against the skill rather than against the session's working directory.

The body is passed to the model verbatim; meka does not rewrite anything inside it. Keeping skills free of host-specific placeholders is what lets the same `SKILL.md` run under meka and other Agent Skills hosts unchanged.

## Storage Location

| Platform | Path |
|----------|------|
| Linux | `~/.config/meka/skills/<name>/SKILL.md` (`$XDG_CONFIG_HOME/meka/skills/`) |
| macOS | `~/Library/Application Support/meka/skills/<name>/SKILL.md` |
| Windows | `%APPDATA%\meka\skills\<name>\SKILL.md` |

## How the Agent Uses Skills

When skills are available, the per-turn context includes a `[Skills]` section like:

```
[Skills]

- **download-videos**: Download videos from various websites using yt-dlp. Use when the user wants a video off a URL.
- **deploy-kubernetes**: Deploy services to a K8s cluster. Use when the user asks to deploy to Kubernetes.
```

The list is sent once, not on every turn. Adding, editing, or removing a skill mid-session is picked up on the next prompt and announced as a short note naming just what changed, so a long session doesn't pay for the whole list repeatedly.

Skills are listed in `priority` order, lowest first, with the name breaking ties. The index is capped at 200 entries and 8 KiB; anything past that is replaced by a count and a pointer to `skill_search`, so a large skill store degrades into "search me" rather than silently eating the context window. The `priority` number itself is not rendered: a skill should be invoked because the request matches its stated purpose, not because it outranks another one.

The agent loads a skill by calling the `skill_read` tool:

```
skill_read(name: "download-videos")
```

The tool returns the full body of `SKILL.md` as its output. The agent then follows the instructions.

Whenever a skill body is loaded (by the `skill_read` tool, `--skill`, `/skill`, `agent_spawn`, or `meka skill show`), it is prefixed with a one-line header naming the skill's directory:

```
Base directory for this skill and its bundled files: /home/user/.config/meka/skills/download-videos
```

This is what lets the agent locate files bundled alongside `SKILL.md` when the body refers to them by relative path (e.g. `scripts/helper.sh`).

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
  Note that `meka skill add --force` is a *replace*, not an update: it deletes the skill directory
  and rebuilds it from the template, taking the body and any bundled files with it.
- Skills the agent *creates* are stamped `author: meka (agent-authored)`, so `meka skill list` shows
  where an entry came from. An existing `author` is kept, so an agent refining a skill you wrote
  does not reassign it to itself. Informational, not a guard.
- A file that exists at that name but is not a valid skill is **refused, not overwritten**. Such a
  file is invisible everywhere else in meka, so nothing could tell you what was about to be lost.
- **Skills declaring a `source_url` are refused** by both tools. Those are upstream-managed, and
  `meka skill update` would silently revert any change, so an edit there is futile rather than
  merely risky. A hand-written skill with no `source_url` is not protected; the flag being off by
  default, and your config directory being in version control, is the safety net for that case.
- `skill_delete` removes the whole skill directory, including any bundled files, matching
  `meka skill remove`.

## Updating Skills

A skill that declares a `source_url` can be re-fetched and replaced on disk with `meka skill update`:

```bash
meka skill update download-videos   # update one skill
meka skill update --all             # dry run: lists what would update
meka skill update --all --yes       # apply the updates
```

`source_url` should be an `https://` link to a raw `SKILL.md` (e.g. a GitHub raw URL or a gist raw URL). The fetch is validated (the response must parse as a valid skill) before the on-disk file is atomically replaced, so a 404 page or a malformed file leaves the existing skill untouched. If the fetched content is byte-identical to what's on disk, nothing is written.

`meka skill update --all` without `--yes` is a dry run: it lists every skill that would be updated and applies nothing. This is the confirmation gate for a bulk remote fetch; re-run with `--yes` to apply.

Only the `SKILL.md` file is fetched. Helper scripts bundled alongside it in the skill directory are **not** updated this way; `source_url`-based update is for single-file skills.

> **Trust note.** A skill body is a set of instructions the agent follows. `meka skill update` replaces that content with whatever the `source_url` currently serves. Review the source you point it at, and prefer `--all` (with its dry-run default) over blind updates.

## Tips

- Use short, unambiguous skill names (e.g. `setup-postgres`, not `pg`). The name is what the agent sees and calls.
- Write `description` concisely, and fold the "use when..." trigger into it. It is sent to the model and consumes tokens.
- Keep each skill focused on a single topic or procedure. Spawn multiple skills rather than one giant one.
- Bundle supporting files in the skill directory and reference them by relative path (`scripts/file.ext`).
- Skills are re-discovered on every prompt, so you can add, edit, or remove skills mid-session without restarting meka.
