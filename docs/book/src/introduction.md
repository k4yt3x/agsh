# Introduction

**meka** is a general-purpose AI agent harness, the layer that wraps a large language model with everything it needs to act as an autonomous agent: a tool set, working memory, context management, persistent sessions, a permission model, and several ways to drive it. You bring a model (Claude or OpenAI, API key or subscription); meka turns it into an agent that can read and edit files, run commands, search the web, call MCP servers, and delegate to sub-agents to get real work done.

The name reflects the design: the model is the pilot, and meka is the mech it operates. The pilot (provider/model) is swappable; the harness around it stays the same.

```text
meka [r] > find all Rust files in this project and count the lines of code
```

You describe the goal in natural language and the agent decides which tools to use to reach it.

## Use It However You Work

The same agent core is exposed through four front-ends:

- **Interactive**: a permission-gated REPL for conversational work in your terminal
- **One-shot**: `meka "..."` for scripts, pipelines, and CI
- **Editor (ACP)**: run as an [Agent Client Protocol](https://agentclientprotocol.com/) agent inside editors like Zed
- **HTTP service**: [`meka serve`](./usage/http-api.md) exposes the agent over HTTP+JSON for bots, web UIs, and other programs

## What the Harness Provides

- **Built-in tools**: file read/write/edit, glob search, regex content search (ripgrep), web fetch, web search, and shell command execution
- **Pluggable providers**: OpenAI API, OpenAI Codex (ChatGPT subscription), Claude API, Claude OAuth (Claude subscription), and any OpenAI-compatible endpoint
- **MCP support**: extend the agent with tools, resources, and prompts from external MCP servers
- **Permission model**: control what the agent can do (none/read/ask/write), switchable mid-session
- **Sessions**: conversations persisted in SQLite; resume, export, or compact any session
- **Working memory**: a session-scoped scratchpad for intermediate results that stays out of the context window
- **Sub-agents**: delegate research or analysis to sub-agents that can orchestrate their own sub-agent teams and be run at a restricted permission level
- **Skills**: load reusable, user-authored instruction packages on demand
- **Context management**: automatic compaction keeps long sessions under the model's context limit
- **Extended thinking**: `anthropic-messages` and `claude-subscription` support extended thinking for complex reasoning

## How It Works

1. You give meka a goal in natural language, interactively, one-shot, over ACP, or over HTTP.
2. meka sends it to the configured model along with a system prompt, the tool schemas, and a context block describing the current permission level, working directory, tools, and skills.
3. The model decides which tools to call (if any) and returns text and/or tool calls.
4. meka enforces the current permission level, executes the tool calls, and feeds the results back to the model.
5. The loop repeats until the model is done; the final response is returned (streamed as Markdown in the terminal).
