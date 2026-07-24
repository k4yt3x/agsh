# Quick Start

## 1. Add a Provider

Before the first run, configure a provider profile. `meka provider add` runs the right credential
flow (OAuth login or API-key prompt) and writes the profile to `~/.config/meka/config.toml`:

```bash
# Claude Code subscription (OAuth)
meka provider add work --type claude-oauth --model claude-opus-5

# or a Claude API key
meka provider add work --type claude-api --model claude-opus-5

# or OpenAI
meka provider add work --type openai-api --model gpt-4o
```

`add` prompts for any of `--type` / `--model` you omit, acquires the secret (browser OAuth for
`claude-oauth` / `openai-codex`, an API-key prompt otherwise), stores it in the database, and makes
the profile the default. Add more profiles later and switch with `meka provider use <name>` or the
per-run `--provider <name>` flag.

> If you launch `meka` with no provider configured, it errors and tells you to run `meka provider add`.
> See [Configuration](../configuration/overview.md) for all options and the full `meka provider` reference.

## 2. Start Using meka

After setup, you will see a prompt:

```text
meka [r] >
```

You will see a prompt:

```text
meka [r] >
```

The `[r]` indicates **read** permission mode (the default). The agent can read files and search, but cannot write files or run commands.

## 3. Ask It Something

```text
meka [r] > what files are in the current directory?
```

The agent will use the `find_files` tool to list files and describe them.

## 4. Enable Write Mode

Press **Shift+Tab** to cycle the permission to write mode:

```text
meka [w] >
```

Now the agent can execute commands and modify files:

```text
meka [w] > create a file called hello.txt with the text "hello world"
```

## 5. One-Shot Mode

For quick tasks without entering the interactive shell:

```bash
meka "what is my current working directory?"
```

The process exits after the agent responds.

## 6. Continue a Previous Session

To pick up where you left off, continue the last session:

```bash
meka -c
```

Or resume a specific session by its UUID:

```bash
meka -c 550e8400-e29b-41d4-a716-446655440000
```

See [Sessions](../usage/sessions.md) for more details.
