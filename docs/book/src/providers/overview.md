# Providers Overview

Providers are the LLM inference backends that meka uses to process your instructions. meka ships with four built-in backends, each selectable as a profile `type`:

| Backend | Auth | API | Notes |
|---------|------|-----|-------|
| [`openai-api`](./openai-api.md) | API key | Chat Completions | Works with OpenAI and any compatible endpoint (Ollama, vLLM, OpenRouter, …) |
| [`openai-codex`](./openai-codex.md) | OAuth login | OpenAI Responses | Uses a ChatGPT subscription; talks to `chatgpt.com/backend-api/codex` like the Codex CLI |
| [`claude-api`](./claude-api.md) | API key | Claude Messages | Direct Claude API, billed per-token |
| [`claude-oauth`](./claude-oauth.md) | OAuth login | Claude Messages | Uses a Claude Code subscription; replicates Claude Code's request shape and attestation |

## Configuring a Provider

Providers are configured as named profiles. The easiest way is `meka provider add`, which writes the
profile to the config file and stores the secret (API key or OAuth token) in the database:

```console
$ meka provider add work --type claude-oauth --model claude-opus-5
```

This produces a `[providers.work]` entry in `~/.config/meka/config.toml`:

```toml
default_provider = "work"

[providers.work]
type  = "claude-oauth"
model = "claude-opus-5"
```

## Selecting a Provider

meka uses the profile named by `--provider <name>` (per run), else `default_provider`, else the sole
profile. Switch the default with `meka provider use <name>`:

```bash
meka --provider work     # this run only
meka provider use work   # persist as default_provider
```

There is no environment-variable override for provider selection.

## OpenAI-Compatible APIs

The `openai-api` backend works with any API that implements the OpenAI Chat Completions format. This includes:

- **OpenAI** (default endpoint)
- **Ollama** (`http://localhost:11434/v1`)
- **OpenRouter** (`https://openrouter.ai/api/v1`)
- **vLLM**, **LiteLLM**, and other OpenAI-compatible servers

Set the profile's `base_url` (or the `--base-url` flag for one run) to point at the alternative endpoint.

## claude-api vs claude-oauth

Both talk to Claude's `/v1/messages` endpoint, but the auth and request shape differ:

- **`claude-api`** is the straightforward path: an `x-api-key` header, a plain system prompt, no extra headers. Choose this when you have a Claude API key.
- **`claude-oauth`** replicates the Claude Code CLI exactly: OAuth tokens, fingerprint-encoded version header, xxHash64 attestation over the request body, injected billing system block. Choose this when you want to use a Claude Code subscription. Any deviation from the expected shape causes requests to be rejected, so avoid proxies that rewrite headers or reformat the body.

## openai-api vs openai-codex

The two OpenAI-flavoured providers hit different endpoints with different protocols:

- **`openai-api`** posts to `/chat/completions` on `api.openai.com` (or any compatible endpoint), authenticating with an API key. This is the right choice when you have an OpenAI billing account or are pointing at a self-hosted OpenAI-compatible server.
- **`openai-codex`** posts to `chatgpt.com/backend-api/codex/responses` using the **OpenAI Responses API** (a different protocol: different request body shape, different streaming events). Authentication is OAuth against `auth.openai.com`, mirroring the first-party Codex CLI. Choose this to use a ChatGPT Plus / Pro / Team / Business subscription instead of a per-token API key.

## Streaming vs Non-Streaming

By default, meka uses streaming mode: tokens appear in the terminal as they are generated. Use `--no-stream` to wait for the complete response before displaying it.

Streaming is recommended for interactive use. Non-streaming may be useful for scripting or when the provider does not support SSE.
