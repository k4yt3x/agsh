# meka

A general-purpose AI agent harness.

> [!CAUTION]
> Agents can perform potentially destructive actions. Exercise caution when granting a permission mode that can modify files or run commands.

> [!IMPORTANT]
> meka is opinionated software and has not stabilized. Defaults, configuration keys, tool names, and stored formats change between releases. Read the changelog before upgrading.

![meka Screenshot](https://github.com/user-attachments/assets/2efa1688-1461-4d26-9743-a3e88203e522)

## Overview

meka wraps a large language model with a tool set, durable memory, persistent sessions, and a permission model. Bring a provider and it becomes an agent that reads and edits files, runs commands, searches the web, calls MCP servers, and delegates to sub-agents.

Supported providers:

- **Anthropic Messages**: your own API key; also reaches LiteLLM, Ollama and anything else speaking it.
- **OpenAI Chat Completions**: your own API key, against any endpoint serving that format.
- **OpenAI Responses**: OpenAI's newer protocol, likewise against any server that offers it.
- **Claude subscription** / **ChatGPT subscription**: sign in with a subscription instead of a key.

## Installation

Download a pre-built binary from [GitHub Releases](https://github.com/k4yt3x/meka/releases/latest), or install with Cargo:

```bash
cargo install --locked --git https://github.com/k4yt3x/meka.git
```

Building from source needs Rust 1.95 or newer and a C toolchain, which `rusqlite` uses to compile the bundled SQLite.

Tagged releases also publish a container image, which the [`mekabox`](contrib/container/mekabox) wrapper uses to run the agent unrestricted against a disposable filesystem:

```bash
docker run --rm -it ghcr.io/k4yt3x/meka:latest --help
```

## Quick Start

Add a provider profile with `meka provider add`. It runs the OAuth login (or prompts for an API key), stores the secret in the database, and writes the profile to `~/.config/meka/config.toml`:

```bash
meka provider add work --type claude-subscription --model claude-opus-5
```

A profile pins a backend and a model. The backend is either a wire protocol (`anthropic-messages`, `openai-chat-completions`, `openai-responses`) or a subscription account (`claude-subscription`, `chatgpt-subscription`). Add several and switch with `meka provider use <name>` or `--provider <name>`. For an OpenAI-compatible endpoint like OpenRouter, set `--base-url`:

```bash
meka provider add openrouter --type openai-chat-completions --model anthropic/claude-opus-5 \
    --base-url https://openrouter.ai/api/v1
```

Run `meka` and start typing. Press Shift+Tab to cycle permissions (none, read, workspace, unrestricted):

```console
meka [r] > find all TODO comments in this project
meka [u] > install and start nginx
```

See the [documentation](https://docs.meka.so) for the full usage guide.

## Features

- **Skills**: [Agent Skills](https://agentskills.io/specification) compliant, so skills work across clients.
- **Memory**: notes the agent keeps for itself, carried into every later session.
- **MCP**: add tools, resources, and prompts from any MCP server.
- **Sub-agents**: delegate work to children that never exceed your permission level.
- **Sandboxed shell**: read mode confines commands using the OS's own sandbox.
- **Scheduling**: have the agent prompt itself later, once or on a cron.
- **Background tasks**: long jobs run detached and report back when done.
- **Sessions**: resume, fork, rewind, or export any past conversation.
- **Context management**: compacts itself before the window fills.
- **Standing instructions**: your own rules, applied to every session.

## Interfaces

The same agent core is available through several interfaces:

- **CLI**: an interactive REPL, or one-shot commands for scripts.
- **ACP**: runs inside editors like Zed via the [Agent Client Protocol](https://agentclientprotocol.com/).
- **HTTP API**: embed meka in your own apps and bots.

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
- `skill_*`: load, search, and optionally author skills
- `schedule_*`: run a prompt later, once or on a cron
- `task_list` / `task_cancel`: manage work the agent detached to the background
- `render_image`: render an image into the conversation for vision models
- `mcp_resource_*` / `mcp_prompt_*`: read or render content from MCP servers
- `load_tool`: fetch the full schema for a tool held back to keep the prompt small

Run `meka tools list` for the current set with descriptions. Long-output tools take an optional `scratchpad` parameter to save their output there instead of returning it.

## Permissions

The prompt indicator shows the current permission mode. Press **Shift+Tab** to cycle between modes:

- `[n]` **none**: no tools; the model can only reply with text
- `[r]` **read**: read-only tools, and a shell sandboxed against writes
- `[w]` **workspace**: every tool; writes confined to the cwd and any `--writable-root`
- `[a]` **ask**: every tool, each call approved by you; enable it under `[permissions]`
- `[u]` **unrestricted**: every tool, with no boundary on where writes land

## Sessions

Conversations are persisted in a local SQLite database and can be resumed:

- `meka -c` continues the last session
- `meka -r <UUID>` resumes a session by UUID, or by a leading prefix of one
- `meka session list` / `delete` / `export` manage and export past sessions
- `/compact`, `/fork`, `/rewind`, `/export` act on the current session from the shell

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
