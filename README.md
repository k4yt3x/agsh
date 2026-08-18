# meka

A general-purpose AI agent harness.

> [!CAUTION]
> Agents can perform potentially destructive actions. Exercise caution when granting write permissions. It is not recommended to run meka on important systems with write permissions enabled.

> [!WARNING]
> Agents can consume a large number of tokens on complex tasks. If you're on a subscription, be prepared for quota exhaustion; if you are billed per token, it is recommended that you set a spending limit on the API key.

![meka Screenshot](https://github.com/user-attachments/assets/e94c40ee-76ae-4b00-bcfe-1c1d9d075a2b)

## Overview

meka is a general-purpose AI agent harness: it wraps a large language model with a tool set, working memory, persistent sessions, a permission model, and several front-ends (an interactive REPL, one-shot commands, an editor agent via ACP, and an HTTP service). Bring a provider (Claude or OpenAI, API key or subscription) and meka turns it into an agent that reads and edits files, runs commands, searches the web, calls MCP servers, and delegates to sub-agents to accomplish real tasks.

Supported providers:

- **OpenAI API**: bring your own API key. Also works with any OpenAI-compatible endpoint.
- **OpenAI Codex**: authenticate with a ChatGPT subscription.
- **Claude API**: bring your own API key.
- **Claude OAuth**: authenticate with a Claude subscription.

## Installation

Download a pre-built binary from [GitHub Releases](https://github.com/k4yt3x/meka/releases/latest), or install with Cargo:

```bash
cargo install --locked --git https://github.com/k4yt3x/meka.git
```

Building from source needs Rust 1.95 or newer and a C toolchain, which `rusqlite` uses to compile the bundled SQLite.

## Quick Start

Add a provider profile with `meka provider add`. It runs the OAuth login (or prompts for an API key), stores the secret in the database, and writes the profile to `~/.config/meka/config.toml`:

```bash
meka provider add work --type claude-oauth --model claude-opus-4-6
```

The profile pins a backend `type` (`openai-api`, `openai-codex`, `claude-api`, or `claude-oauth`) plus a model. Add several profiles and switch with `meka provider use <name>` or `--provider <name>`. For an OpenAI-compatible endpoint like OpenRouter, set `--base-url`:

```bash
meka provider add openrouter --type openai-api --model anthropic/claude-opus-4.6 \
    --base-url https://openrouter.ai/api/v1
```

Run `meka` and start typing. Press Shift+Tab to cycle permissions (none, read, ask, write):

```console
meka [r] > find all TODO comments in this project
meka [w] > install and start nginx
```

See the [documentation](https://docs.meka.so) for the full usage guide.

## Interfaces

The same agent core is available through several interfaces:

- **CLI REPL**: an interactive prompt in your terminal.
- **ACP**: makes meka work inside editors like Zed via the [Agent Client Protocol](https://agentclientprotocol.com/).
- **HTTP API**: embed meka as an agent backend in your own apps, bots, and services.

## Tools

The agent has access to the following built-in tools:

- `execute_command`: run commands and read their output
- `read_file` / `write_file` / `edit_file`: read, create, and modify files
- `find_files`: find files by name or glob pattern
- `search_contents`: search file contents with regex, powered by ripgrep
- `fetch_url`: fetch a web page as markdown
- `search_web`: search the web for current information
- `scratchpad_*`: session-scoped working memory for intermediate results
- `todo`: structured task tracking, with live progress display
- `memory_*`: notes that survive the session, loaded into every later one
- `conversation_read` / `conversation_search`: re-read this session's history
- `context_check` / `context_compact`: read the remaining window, or compact on purpose
- `agent_*`: delegate to a sub-agent, which never exceeds your permission level
- `skill_*`: load, search, and optionally author reusable prompt templates
- `schedule_*`: run a prompt later, once or on a cron
- `task_list` / `task_cancel`: manage work the agent detached to the background
- `render_image`: render an image into the conversation for vision models
- `mcp_resource_*` / `mcp_prompt_*`: read or render content from MCP servers

Run `meka tools list` for the current set with descriptions.

Long-output tools take an optional `scratchpad` parameter to save their output there instead.

## Permissions

The prompt indicator shows the current permission mode. Press **Shift+Tab** to cycle between modes:

- `[n]` **none**: no tools available, text-only responses
- `[r]` **read**: read-only tools (file reading, searching, web, sandboxed shell)
- `[a]` **ask**: all tools available, but each call requires user approval
- `[w]` **write**: all tools enabled, including shell execution and file writes

## Sessions

Conversations are persisted in a local SQLite database and can be resumed:

- `meka -c` continues the last session
- `meka -r <UUID>` resumes a specific session by UUID, or by a leading prefix of one
- `meka session list` lists past sessions
- `meka session delete <UUID>` deletes sessions
- `meka session export <UUID>` exports a session as Markdown
- `meka history list` / `meka history clear` view or clear REPL input history
- `/export` exports the current session from within the shell
- `/compact` summarizes and compacts the session history

## Features

- **Extended thinking**: on by default for Claude; `thinking` picks adaptive, budgeted, or off
- **Syntax-highlighted output**: markdown rendering with syntect-highlighted code blocks, using bat's Monokai Extended theme
- **Auto-compact**: automatically compacts the conversation when approaching the context limit
- **MCP support**: extend the agent with tools from external MCP servers
- **Skills**: load reusable prompt templates from `~/.config/meka/skills/`

## Shell Escape

Prefix input with `!` to execute a command directly, bypassing the LLM:

```console
meka [r] > !uname -a
meka [r] > !docker ps
```

Type `exit`, `quit`, or press **Ctrl+D** to leave the shell.

## AI Use Declaration

AI tools were used to assist the design and implementation of this project. All design decisions were made by humans, and every change was reviewed and approved by a human maintainer.

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT).\
Copyright 2026 K4YT3X.
