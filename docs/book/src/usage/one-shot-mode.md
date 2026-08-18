# One-Shot Mode

One-shot mode runs a single prompt and exits, similar to `bash -c`. It takes `--oneshot`:

```bash
meka --oneshot "your prompt here"
```

The agent processes the prompt (including any tool calls), prints its response, and the process terminates. The session UUID is printed to stderr on exit.

A prompt **without** `--oneshot` is not a one-shot run: it seeds the first turn and then leaves you at the REPL prompt, which is the right default when you are working interactively and the first thing you want is already in your shell history.

`--oneshot` requires something to do, so it needs a prompt argument or `--skill`.

An empty or whitespace-only prompt is rejected rather than sent.

`--permission ask` has nothing to ask from here: there is no prompt to answer, so every tool that needs approval is refused. meka says so once at startup and names each tool as it is refused, but the run is still less useful than it looks. Use `read` or `write` for a non-interactive run, or [`meka serve`](./http-api.md) if you need a human in the loop over an API.

## Examples

```bash
# Simple question
meka --oneshot "what is my current working directory?"

# File operations (requires write permission)
meka --oneshot --permission write "create a file called notes.txt with today's date"

# Search
meka --oneshot "find all TODO comments in this project"

# Web search
meka --oneshot "search the web for the latest Rust release"
```

## Combining with Other Flags

All configuration flags work in one-shot mode:

```bash
# Use a specific provider and model
meka --oneshot --provider work -m claude-sonnet-4-20250514 "explain this codebase"

# With write permission
meka --oneshot --permission write "run 'cargo test' and summarize the results"

# Disable streaming
meka --oneshot --no-stream "read README.md and summarize it"

# Run one turn against an existing session
meka --oneshot -r 550e8400 "summarise what we decided"
```

## Session Behavior

One-shot mode creates a new session for each invocation, unless you point it at an existing one with `-c` (most recent) or `-r <SESSION>` (specific). Those run a single turn against that conversation and exit, which is the usual shape for scripting against a session built up earlier.

The session UUID is printed to stderr when the run completes:

```text
Session: 550e8400-e29b-41d4-a716-446655440000
```

You can resume this session later in interactive mode:

```bash
meka -r 550e8400-e29b-41d4-a716-446655440000
```
