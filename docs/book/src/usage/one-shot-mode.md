# One-Shot Mode

One-shot mode runs a single prompt and exits, similar to `bash -c`:

```bash
meka "your prompt here"
```

The agent processes the prompt (including any tool calls), prints its response, and the process terminates. The session UUID is printed to stderr on exit.

## Examples

```bash
# Simple question
meka "what is my current working directory?"

# File operations (requires write permission)
meka --permission write "create a file called notes.txt with today's date"

# Search
meka "find all TODO comments in this project"

# Web search
meka "search the web for the latest Rust release"
```

## Combining with Other Flags

All configuration flags work in one-shot mode:

```bash
# Use a specific provider and model
meka --provider claude-oauth -m claude-sonnet-4-20250514 "explain this codebase"

# With write permission
meka --permission write "run 'cargo test' and summarize the results"

# Disable streaming
meka --no-stream "read README.md and summarize it"

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
meka -s 550e8400-e29b-41d4-a716-446655440000
```
