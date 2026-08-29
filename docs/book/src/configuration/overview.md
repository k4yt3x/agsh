# Configuration Overview

meka is configured with named **provider profiles** in a config file at
`~/.config/meka/config.toml`, plus secrets stored in the database. The quickest way to get started
is to let `meka provider add` write both for you:

```console
$ meka provider add work --type claude-subscription --model claude-opus-5
```

That command writes a `[providers.work]` profile to the config file, runs the OAuth login (or prompts
for an API key, depending on the backend), stores the secret in the database, and makes the profile
the default. The resulting config looks like:

```toml
default_provider = "work"

[providers.work]
type  = "claude-subscription"
model = "claude-opus-5"
```

See [Config File](./config-file.md) for the full reference and the [`meka provider`](./config-file.md#meka-provider-cli) command suite.

## Required Settings

To run a turn, meka needs an active provider profile that pins a backend `type` and `model`, and a
stored credential for it. If no profile can be selected, or the active profile has no model or no
credential, meka prints an error pointing at `meka provider add` / `meka provider login`.

| Setting | Source | Named on the command line |
|---------|--------|---------------------------|
| Profile for an existing session | The session's own row | `--provider <name>`, which **repins** the row |
| Profile for a new session | `default_provider` in config, or the sole profile | `--provider <name>` |
| Backend (`type`) | `[providers.<name>].type` | -- |
| Model | `[providers.<name>].model` | -- |
| Every other model-tied setting | `[providers.<name>].*` | -- |
| Credential (API key / OAuth) | Database, via `meka provider add` / `login` | -- |

## A profile is indivisible

A profile is a named bundle: the backend, the endpoint, the credential keyed to it, the model, and
every model-tied setting (`context_window`, `vision`, `max_output_tokens`, `effort`, `thinking`,
`thinking_budget`, `redact_thinking`). A session selects one by name and records that name.
**Nothing overrides a field inside one.**

There is deliberately no `--model`, `--base-url`, `--thinking` or `--thinking-budget`. A flag that
moved one field of the bundle left the rest behind, so a session could run a 200K model while
gauging its context against the 1M window its profile still stated, and never auto-compact.

To change a setting, edit the profile:

```bash
meka provider set work model claude-opus-5
```

To run something different, make a second profile and select it:

```bash
meka provider add fast --type anthropic-messages --model claude-haiku-4-5 --context-window 200000
meka --provider fast "quick question"
```

## Override Layers

Provider configuration is layered as follows; higher-priority layers override lower ones:

1. **The session's own row**: the profile it was created with. A session that exists runs on what
   its row says, whatever `default_provider` later becomes.
2. **`--provider <name>`**: on a new session this chooses what the row records; on a resume it
   **rewrites** the row, so the change holds for every later turn and from every surface. See
   [what a resume restores](../usage/sessions.md#what-a-resume-restores).
3. **Config file**: persistent profiles in `~/.config/meka/config.toml`.
4. **Built-in defaults**: permission defaults to `read`, streaming defaults to on.

There is **no environment-variable tier** for provider configuration; an ambient `OPENAI_API_KEY` or
`MEKA_PROVIDER` has no effect (see [Environment Variables](./environment-variables.md)).

## Credential Resolution

The credential for the active profile is loaded from the database, keyed by the profile name. It is
acquired interactively:

- `meka provider add <name>` runs the OAuth login (`claude-subscription`, `chatgpt-subscription`) or prompts for the
  API key (`anthropic-messages`, `openai-chat-completions`, `openai-responses`) when the profile is created.
- `meka provider login <name>` re-acquires it for an existing profile (rotate an API key, recover
  from a dead OAuth refresh token), keeping every other setting on the profile. Add
  `--api-key-stdin` to pipe the key in for scripted rotation.
- `meka provider remove <name>` deletes the stored credential and the profile.

Because secrets are keyed per profile, two profiles using the same backend (for example, two Claude
accounts) keep independent credentials.

Deleting a `[providers.<name>]` block by hand removes the settings but not the secret, which stays in
the database under that name. `meka provider list` names any credential left that way, and `meka
provider remove <name>` deletes it; see [Leftover
credentials](./config-file.md#leftover-credentials).

## Why some settings have no config key

A few things are deliberately CLI-only, with no `config.toml` key and no environment variable.
`--writable-root` is the current example: which folders a run may write at `workspace` permission is
a per-run scope, like the working directory itself, not a preference worth persisting. Writing it
into a file would make the boundary depend on where the file lives rather than on what you asked for
this time.

This is the same reasoning that keeps the working directory out of config, and it is the exception
to "config.toml is the complete source of truth": that rule covers persistent *settings*, and a
per-run scope is not one.


## When edits take effect

`meka` in the terminal reads `config.toml` and your instructions files at startup, so anything you
change applies from the next command. A long-lived host is different: `meka serve` and `meka acp`
read both **once**, when the process starts, and keep what they read for as long as they run.

Two consequences worth knowing:

- **`meka provider add` while a server is running does not reach it.** The new profile is on disk
  and `meka provider list` shows it, but `POST /v1/sessions` and ACP's provider picker answer "not
  configured" until the server is restarted. The same applies to editing an existing profile.
- **Editing your instructions files does not reach it either.** They are read once and go into the
  cached prompt prefix that every session shares.

Restart the server to pick either up. Everything else follows a live source and needs no restart:
skills are re-read per turn, memories per turn, and MCP tool lists follow the server.

A **rotated credential** sits between the two, and the distinction matters if you are rotating
because a key leaked. `meka provider login <name>` from a second process is picked up without a
restart by anything that builds a provider *after* it: newly created sessions, ones the server
re-attaches after eviction, and ones explicitly repinned by `PATCH /v1/sessions/{id}`,
`session/set_config_option` or `/provider`.

A session already resident in memory holds the provider it was built with. For an **API-key**
profile that means it keeps presenting the old key until it is evicted (`[serve] idle_timeout`, 24
hours by default) or the server restarts. For the **OAuth** backends (`claude-subscription`,
`chatgpt-subscription`) the live provider re-reads the stored bundle when it next refreshes its
token, so a rotation is usually adopted sooner, but nothing makes that happen on demand.

**To be certain a revoked credential is out of use, restart the host.**
