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

| Setting | Source | Per-run override |
|---------|--------|------------------|
| Active profile | `default_provider` in config, or the sole profile | `--provider <name>` |
| Backend (`type`) | `[providers.<name>].type` | -- |
| Model | `[providers.<name>].model` | `-m`, `--model` |
| Credential (API key / OAuth) | Database, via `meka provider add` / `login` | -- |

## Override Layers

Provider configuration is layered as follows; higher-priority layers override lower ones:

1. **CLI flags**: per-invocation overrides (`--provider`, `--model`, `--base-url`).
2. **Config file**: persistent profiles in `~/.config/meka/config.toml`.
3. **Built-in defaults**: permission defaults to `read`, streaming defaults to on.

For example, `--model gpt-4o-mini` on the command line overrides the active profile's `model` for
that run. There is **no environment-variable tier** for provider configuration; an ambient
`OPENAI_API_KEY` or `MEKA_PROVIDER` has no effect (see [Environment Variables](./environment-variables.md)).

## Credential Resolution

The credential for the active profile is loaded from the database, keyed by the profile name. It is
acquired interactively:

- `meka provider add <name>` runs the OAuth login (`claude-subscription`, `chatgpt-subscription`) or prompts for the
  API key (`anthropic-messages`, `openai-chat-completions`, `openai-responses`) when the profile is created.
- `meka provider login <name>` re-acquires it for an existing profile (rotate an API key, recover
  from a dead OAuth refresh token).
- `meka provider remove <name>` deletes the stored credential and the profile.

Because secrets are keyed per profile, two profiles using the same backend (for example, two Claude
accounts) keep independent credentials.

Deleting a `[providers.<name>]` block by hand removes the settings but not the secret, which stays in
the database under that name. `meka provider list` names any credential left that way, and `meka
provider remove <name>` deletes it; see [Leftover
credentials](./config-file.md#leftover-credentials).
