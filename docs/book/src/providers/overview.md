# Providers Overview

Providers are the LLM inference backends meka uses to run your instructions. meka ships with five, each selectable as a profile `type`:

| Backend | Protocol | Endpoint | Auth |
|---------|----------|----------|------|
| [`anthropic-messages`](./anthropic-messages.md) | Anthropic Messages | `{base}/v1/messages` | API key |
| [`claude-subscription`](./claude-subscription.md) | Anthropic Messages | `api.anthropic.com/v1/messages` | Claude subscription |
| [`openai-chat-completions`](./openai-chat-completions.md) | OpenAI Chat Completions | `{base}/chat/completions` | API key |
| [`openai-responses`](./openai-responses.md) | OpenAI Responses | `{base}/responses` | API key |
| [`chatgpt-subscription`](./chatgpt-subscription.md) | OpenAI Responses | `chatgpt.com/backend-api/codex/responses` | ChatGPT subscription |

**A backend names the wire protocol, not a vendor.** That is deliberate, and it cuts both ways. One vendor can serve several protocols: OpenAI publishes Chat Completions *and* Responses, and they are different request shapes, not options on one. One protocol is served by many vendors: `/v1/messages` is implemented by Anthropic, Amazon Bedrock, Databricks, LiteLLM and Ollama, so calling it "the Claude API" would misname it the moment you point it elsewhere.

The two subscription backends are the exception, and carry a vendor name instead. What you pick there is a billing relationship; the endpoint and the client shape come with it and are not yours to choose.

Synthetic is the clearest case for why this matters. One vendor, two protocols, two base URLs:

```toml
[providers.synthetic-claude]
type     = "anthropic-messages"
base_url = "https://api.synthetic.new/anthropic/v1"

[providers.synthetic-gpt]
type     = "openai-chat-completions"
base_url = "https://api.synthetic.new/openai/v1"
```

## Configuring a Provider

Providers are configured as named profiles. The easiest way is `meka provider add`, which writes the
profile to the config file and stores the secret (API key or OAuth token) in the database:

```console
$ meka provider add work --type claude-subscription --model claude-opus-5
```

This produces a `[providers.work]` entry in `~/.config/meka/config.toml`:

```toml
default_provider = "work"

[providers.work]
type  = "claude-subscription"
model = "claude-opus-5"
```

## Selecting a Provider

A **new** session runs on the profile named by `--provider <name>`, else `default_provider`, else
the sole profile. Switch the default with `meka provider use <name>`:

```bash
meka --provider work     # pick the profile this session starts on
meka provider use work   # persist as default_provider
```

There is no environment-variable override for provider selection.

A **resumed** session ignores all three and runs on the profile it recorded, so `meka -c` stays
where the conversation was had whatever `default_provider` currently says. `--provider` on a resume
is not a per-run override either: it **repins** the session, rewriting the row so every later resume
keeps it. `meka session list` shows what each session runs on, and `--long` adds any model or
endpoint override on top. You can also move a live session with `/provider <name>` in the REPL,
`PATCH /v1/sessions/{id}` over HTTP, or the Provider picker in an ACP client. See
[Sessions](../usage/sessions.md#what-a-resume-restores).

## Pointing a backend somewhere else

Every API-key backend takes a `base_url` (or `--base-url`, which is recorded on the session the way `--provider` is), so the protocol you pick is independent of who serves it:

| Server | Chat Completions | Responses | Anthropic Messages |
|--------|------------------|-----------|--------------------|
| OpenAI | yes | yes | no |
| Anthropic | no | no | yes |
| Ollama | yes | yes (v0.13.3+) | yes |
| OpenRouter | yes | yes (beta) | yes |
| vLLM / LM Studio | yes | yes | no |
| Synthetic | yes | no | yes |

Where a server offers both OpenAI protocols, prefer [`openai-responses`](./openai-responses.md): it is what OpenAI recommends for new work and what the agent tooling ecosystem has moved to. Use `openai-chat-completions` for a server that does not serve Responses.

Note that several of these also expose a **legacy `/v1/completions`** endpoint. That is a third, different protocol: a bare `prompt` string in, `choices[].text` out, no tool calling. meka does not speak it. It cannot: the agent loop needs tool calls, which that protocol has no representation for.

## anthropic-messages vs claude-subscription

Both talk to Claude's `/v1/messages` endpoint, but the auth and request shape differ:

- **`anthropic-messages`** is the straightforward path: an `x-api-key` header and a plain system prompt, plus `anthropic-beta: interleaved-thinking-2025-05-14` whenever thinking is on (the default). Choose this when you have a Claude API key.
- **`claude-subscription`** replicates the Claude Code CLI exactly: OAuth tokens, fingerprint-encoded version header, xxHash64 attestation over the request body, injected billing system block. Choose this when you want to use a Claude Code subscription. Any deviation from the expected shape causes requests to be rejected, so avoid proxies that rewrite headers or reformat the body.

## Choosing between the OpenAI backends

Three backends, two protocols:

- **`openai-chat-completions`** posts to `/chat/completions` with an API key. Choose it for a server that serves only this protocol.
- **`openai-responses`** posts to `/responses` with an API key, the same protocol `chatgpt-subscription` uses. Choose it for OpenAI, or for any server that serves Responses.
- **`chatgpt-subscription`** posts to `chatgpt.com/backend-api/codex/responses`, authenticating by OAuth against `auth.openai.com` and mirroring the first-party Codex CLI. Choose it to bill a ChatGPT Plus / Pro / Team / Business subscription instead of a per-token API key.

The first two differ by protocol; the last two differ only by auth and endpoint.

## Streaming vs Non-Streaming

By default, meka uses streaming mode: tokens appear in the terminal as they are generated. Use `--no-stream` to wait for the complete response before displaying it.

Streaming is recommended for interactive use. Non-streaming may be useful for scripting or when the provider does not support SSE.
