# Instructions

Standing instructions are your own guidance to the agent, applied to every session on this machine. They land in the system prompt under a `## User Instructions` heading, and the model is told to treat them as hard constraints unless they conflict with safety requirements.

Use them for things that are true of your setup rather than of any one task:

- System policies: "Never install Python packages globally with pip. Always use `uv` or a venv."
- Installed tooling worth knowing about: "Poppler is available; use `pdftotext` for PDFs."
- Workflow preferences: "Prefer ripgrep over grep."
- Compliance rules: "Git commits on this system must be gpg-signed."

## Where they live

Instructions are content, not configuration, so they live at a conventional path beside `config.toml` rather than behind a key inside it. Write:

```
~/.config/meka/instructions.md
```

If the set grows, split it into a directory instead. Every `*.md` file is concatenated in lexical order, so a numeric prefix controls the sequence:

```
~/.config/meka/instructions/
├── 00-style.md
├── 10-security.md
└── 20-tooling.md
```

The directory wins when it has content, so splitting a grown `instructions.md` is a rename rather than a migration. An empty `instructions/` falls back to the file rather than blanking your instructions. Under a custom [`MEKA_CONFIG_DIR`](../configuration/environment-variables.md), both paths follow it.

Check what is actually in effect at any time:

```bash
meka instructions show     # the resolved text, plus where it came from
meka instructions path     # the paths meka checks, and whether each exists
```

`show` prints the text on stdout and the source on stderr, so `meka instructions show 2>/dev/null` pipes cleanly.

## Passing them as a string

A file is the right shape on a workstation, but not everywhere. When the channel carrying the value is a string rather than a filesystem, use one of:

| Source | Form | |
|--------|------|---|
| `--instructions` | text | per-run, wins over everything |
| `MEKA_INSTRUCTIONS` | text | |
| `MEKA_INSTRUCTIONS_FILE` | path | |
| `instructions.md` / `instructions/` | file | the default |

Resolution stops at the first one set, in that order.

This matters most for containers. The [`mekabox`](https://github.com/k4yt3x/meka/blob/master/contrib/container/mekabox) wrapper mounts your config directory into the container **read-only** and then replaces the instructions with container-specific ones, which is a single `-e MEKA_INSTRUCTIONS=…`. Requiring a path would mean writing a temp file on the host and bind-mounting it, and the read-only mount means it could not simply write the file where meka looks.

`MEKA_INSTRUCTIONS_FILE` covers the case where a file exists but you do not control where it is mounted, such as a Kubernetes ConfigMap or a Docker secret. It accepts a directory too, since a ConfigMap mounts as a directory of keys, and in that case takes any regular file rather than only `*.md`: a ConfigMap key is often just `instructions`, and a naming choice made in someone else's YAML should not become a startup failure inside a pod.

Setting `MEKA_INSTRUCTIONS=` to the empty string means "no instructions", suppressing the file rather than falling through to it. That is the way to run a container with your host instructions mounted but not applied.

Setting both environment variables is refused at startup. There is no reading under which someone meant both, so resolving one silently would hide the mistake until the agent behaved unexpectedly.

## When they are read

Once, at startup. Editing takes effect on the next launch, not mid-session.

That is deliberate, and it follows from size. The system prompt heads the prompt-cache prefix, so a large instruction set is billed once and served from cache on every later turn. Re-reading it per turn would either invalidate that prefix whenever the file changed, or push the text down into the conversation where it would compete with actual context.

This is the opposite of [skills](./skills.md) and [memory](./memory.md), which do refresh mid-session. They can afford to: both are indexed rather than included in full, and the index is small.

`meka -c` makes restarting cheap when you do edit them.

## Notes

- Empty or whitespace-only instructions are treated as unset.
- Sub-agents do **not** receive them by default. Instructions describe the top-level agent, and a worker handed one task by one of its turns is not that agent; inheriting the persona is how a sub-agent ends up addressing the user as though it were the one they are talking to. The agent can pass `instructions: "inherit"` to [`agent_spawn`](../tools/overview.md#agent_spawn) when a task genuinely needs the project's standing rules, or pass a [skill](./skills.md) when the direction is reusable.
- They apply at every permission level, including `none`, because you wrote them.
- A set larger than roughly 8k tokens logs a warning at startup. It still works, and it is cached, but it occupies that much of every request's window and is usually a surprise rather than a decision.
- An unreadable file in a directory is skipped with a warning rather than hiding the rest of it. A path you named explicitly via `MEKA_INSTRUCTIONS_FILE` is an error instead, since running without guidance you believe you supplied is worse than not starting.
- A directory contributes at most 100 files; past that it is far more likely pointed somewhere unintended than intentional, so the rest are skipped with a warning.
