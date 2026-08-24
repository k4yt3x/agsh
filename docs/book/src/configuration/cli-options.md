# CLI Options

```text
meka [OPTIONS] [PROMPT]
meka <COMMAND>
```

## Commands

### `provider`

Manage provider profiles (add, list, switch, login, remove). `meka provider add` writes a
`[providers.<name>]` profile to `~/.config/meka/config.toml` and stores its secret in the database.

```bash
meka provider add work --type claude-subscription --model claude-opus-5
meka provider list
meka provider use work
meka provider login work
meka provider remove work
```

See the [`meka provider` CLI reference](./config-file.md#meka-provider-cli) for the full flag list.

### `session`

Manage stored sessions: list them, export one as Markdown, or delete them.

```bash
meka session list [-n <LIMIT>]          # default limit: 20
meka session export <SESSION_ID> [-o <PATH>]   # -o - prints to stdout
meka session delete <SESSION_ID>...
meka session delete --older-than-days <DAYS>
meka session delete --all
```

See [Sessions](../usage/sessions.md#exporting-a-session) for details.

### `history`

View or clear the REPL input history that powers Up-arrow / Ctrl+R recall (distinct from saved
sessions and from the `/history` slash command).

```bash
meka history list [-n <LIMIT>]   # default 50; -n 0 shows all
meka history clear
```

## Arguments

### `[PROMPT]`

Run the agent's first turn immediately with this text as the user message, then drop into the interactive REPL for follow-up. Pair with [`--oneshot`](#--oneshot) to exit after the first turn instead of opening the REPL.

```bash
meka "list all files larger than 1MB in the current directory"   # first turn, then REPL
meka --oneshot "list all files larger than 1MB"                  # first turn, then exit
```

When omitted, meka starts the REPL with no initial input.

## Options

### `-c`, `--continue`

Continue the most recently updated session. Takes no value.

```bash
meka -c                     # pick up where you left off
meka -c "and now add tests" # …with an opening prompt
```

Starting fresh when there is no session yet is not an error; meka just begins a new one.

### `-r`, `--resume <SESSION>`

Resume a specific session. Accepts either the full UUID or any unique leading prefix.

```bash
meka -r 550e8400-e29b-41d4-a716-446655440000     # full UUID
meka -r 550e                                     # prefix; works if unique
meka -r 550e "and now add tests"                 # …with an opening prompt
```

Errors if the session does not exist, the prefix matches multiple sessions (with the matching IDs listed for disambiguation), or the session is locked by another meka instance.

`-c` and `-r` are mutually exclusive. Both work with `--oneshot`, which runs a single turn against the session and exits:

```bash
meka --oneshot -r 550e "summarise what we decided"
```

> **Breaking change.** `-c` used to take an optional session ID (`meka -c 550e8400`); that spelling now belongs to `-r`. Because `-c` was the only flag that could swallow the following argument, `meka -c "fix the bug"` read the prompt as a session ID and failed with a confusing error. Passing an ID to `-c` now tells you to use `-r`.

### `--permission <MODE>`

Set the initial permission mode. Accepts `none` (or `n`), `read` (or `r`), `workspace` (or `w`), `ask` (or `a`), `unrestricted` (or `u`).

```bash
meka --permission workspace
meka --permission ask
```

Default: `read`.

### `--writable-root <PATH>`

Add a directory to the workspace, so writes may land there at `workspace` permission. Repeatable.
The working directory is always a root; this adds to it.

```bash
meka --permission workspace --writable-root ../shared-assets --writable-root /srv/build
```

Deliberately a flag rather than a config key: which folders this run may write is a per-run scope,
like the working directory itself, not a preference to persist.

A path that does not resolve at startup is reported as a warning and kept, so a build directory that
does not exist yet becomes a root the moment it does. A path that is not a directory, or a system
directory the sandbox masks, is refused with a warning: neither can be expressed as a boundary by
every backend.

The masked set is the filesystem root itself plus `/proc`, `/dev`, `/sys`, `/run`, `/tmp` and
`/var/tmp`, and `/run/user` and `$XDG_RUNTIME_DIR` as whole subtrees. A root is refused when it *is*
one of these and when it is an **ancestor** of one, but not when it is merely underneath: a root
under one of these is usually fine and refusing it would be a real loss, since
`/run/media/$USER/drive` is an ordinary external disk. The ancestor half is why `--writable-root
/var` is refused, and it is also why `--writable-root ~` is refused on WSL and on minimal window
managers, where `$XDG_RUNTIME_DIR` lives under `$HOME` and so binding `$HOME` would hand the session
bus back. `/tmp` and `/var/tmp` are in the set because
Bubblewrap masks them with a tmpfs and then binds the requested root back over it, last mount
winning: binding `/tmp/work` restores just that directory, but binding `/tmp` restores the entire
host `/tmp` including every X11, D-Bus and tmux socket in it, which is a route straight back out of
the sandbox. The cost is that a session started with `cd /tmp` has no write boundary, which is the
safe direction to fail.

The flag reaches the REPL, one-shot runs, ACP sessions, and isolated scheduled fires under
`meka serve`. It does **not** reach sessions created through `POST /v1/sessions`, which are
single-root by design.

### `-p`, `--provider <NAME>`

Select which configured provider profile to use for this run. Takes the name of a profile from
`[providers.<name>]`, overriding `default_provider` in the config file.

```bash
meka -p work
meka --provider work
```

The value is a profile name (e.g. `work`, `personal`), not a backend type. List configured profiles with `meka provider list`.

### `-m`, `--model <MODEL>`

Override the active profile's model for this run.

```bash
meka -m gpt-5.6-sol
```

### `--base-url <URL>`

Override the active profile's API base URL for this run.

```bash
meka --base-url http://localhost:11434/v1
```

### `--no-stream`

Disable streaming mode. The agent waits for the complete response before displaying it. By default, responses are streamed token-by-token.

```bash
meka --no-stream
```

### `--render-mode <MODE>`

Set the output render mode. Accepts `termimad` (default, or `rich`), `syntect`, or `raw`.

- `syntect`: Syntax-highlighted markdown source, including per-language code blocks. Nothing is reflowed, so a table with long cells runs past the terminal width.
- `termimad`: Rendered markdown, reflowed to the terminal: paragraphs re-wrap, wide tables wrap inside their box, and markers are consumed rather than shown. meka parses the CommonMark itself, so `-`/`+` bullets, `__bold__`, `_italic_`, ordered lists, and links all render. Colours come from the same theme as `syntect`, and fenced code blocks are syntax-highlighted by it.
- `raw`: Raw markdown printed verbatim with aligned tables.

`termimad` is the default: meka's own output is table-heavy (`task_list`, `scratchpad_list`, anything the model tabulates), and those run past the right edge under `syntect`. Pick `syntect` when you want to see the markdown source as the model wrote it.

```bash
meka --render-mode raw
```

Can also be set permanently via `display.render_mode` in the config file.

### `--thinking <MODE>`

Extended thinking mode for this run (`anthropic-messages` and `claude-subscription` providers): `adaptive`,
`budgeted`, or `off`. Overrides the profile's
[`thinking`](./config-file.md#thinking) key, which defaults to `adaptive`.

```bash
meka --thinking off
```

### `--thinking-budget <TOKENS>`

Set the extended thinking token budget. Read only under `--thinking budgeted` (and the profile's
`thinking = "budgeted"`), since the adaptive encoding lets the model set its own budget and `off`
sends none at all.

```bash
meka --thinking-budget 20000
```

### `--instructions <STRING>`

Standing [instructions](../usage/instructions.md) for this run, replacing the `instructions.md` file and both `MEKA_INSTRUCTIONS*` environment variables. Takes the text itself, not a path; use `"$(cat file.md)"` to read one.

```bash
meka --instructions "Be terse. No code fences in answers."
```

### `--skill <NAME>`

Invoke a [skill](../usage/skills.md) as the first turn. Mirrors the REPL slash command [`/skill <name> [extra...]`](../usage/skills.md#invoking-a-skill-from-the-cli). The positional `[PROMPT]` arg, if given, is prepended to the rendered skill body as additional context. Pair with [`--oneshot`](#--oneshot) to exit after the turn instead of opening the REPL.

```bash
meka --skill download-videos "https://example.com/video"             # first turn, then REPL
meka --skill download-videos --oneshot "https://example.com/video"   # first turn, then exit
```

Errors out with a clean message if the skill name is unknown.

### `--oneshot`

Exit after the first turn finishes. Requires either the positional `[PROMPT]` or `--skill <NAME>`; without one of those, meka has nothing to do. Useful for scripts and CI invocations.

```bash
meka --oneshot "summarize the last commit"
meka --oneshot --skill deploy "to staging"
```

### `--eager-load-tool <SERVER:TOOL>`

Eager-load a specific MCP tool for this session, bypassing the `load_tool` round-trip. The tool's schema ships in the cacheable tools-array prefix from turn 1 instead of being deferred. Mirrors the per-server [`eager_load_tools`](./config-file.md#mcpservers) config field: repeatable, raw tool names (the server-advertised form, not `mcp__<server>__<tool>`).

Particularly useful for scripted runs that know up front which tools they'll need. The flag *appends to* whatever `eager_load_tools` lists in `config.toml` for that server; it doesn't replace existing entries. Unknown server names log a warning and are skipped.

```bash
meka --eager-load-tool notion:search --eager-load-tool github:create_issue \
     --oneshot "search Notion for the deploy runbook and open a GitHub issue"
```

### `-v`, `--verbose`

Increase log verbosity. Can be repeated up to three times.

```bash
meka -v      # info
meka -vv     # debug
meka -vvv    # trace
```

### `--help`

Print help information.

### `--version`

Print version information.
