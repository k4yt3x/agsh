# OpenAI Responses

The **Responses API** (`POST {base_url}/responses`) with an API key. This is OpenAI's newer protocol
and the one it recommends for new work; it is also what [`chatgpt-subscription`](./chatgpt-subscription.md)
speaks, so the two differ only in how they authenticate and where they post.

## Setup

```console
$ meka provider add work --type openai-responses --model gpt-5.6-sol
```

```toml
default_provider = "work"

[providers.work]
type  = "openai-responses"
model = "gpt-5.6-sol"
```

## Configuration

### `base_url`

Defaults to `https://api.openai.com/v1`. meka appends `/responses`, so pass the base URL the server
publishes and nothing more:

```toml
base_url = "http://127.0.0.1:11434/v1"      # Ollama
base_url = "https://openrouter.ai/api/v1"   # OpenRouter
```

### `effort`

Maps to `reasoning.effort`. When unset the whole `reasoning` block is omitted and the endpoint
applies its own default. See the [`effort`](../configuration/config-file.md#effort) reference.

## Which servers serve this

| Server | Base URL | Notes |
|--------|----------|-------|
| OpenAI | `https://api.openai.com/v1` | The reference implementation |
| Ollama | `http://127.0.0.1:11434/v1` | **v0.13.3+ only**; earlier versions 404 |
| vLLM | your deployment | |
| LM Studio | your deployment | |
| OpenRouter | `https://openrouter.ai/api/v1` | Beta |
| Synthetic | not served | **Not supported**; use [`openai-chat-completions`](./openai-chat-completions.md) or [`anthropic-messages`](./anthropic-messages.md) |

Only the non-stateful flavour is needed. meka replays the whole conversation every turn and sends
`store: false`, so it never uses `previous_response_id` or server-side conversation state, which is
also all the local runtimes implement.

## Choosing between this and Chat Completions

Both take an API key and both reach most of the same servers, so the question is only which protocol
the server implements and which one you want:

- Prefer **`openai-responses`** where it is available. It is what OpenAI recommends for new work,
  and the agent tooling ecosystem has moved to it: OpenAI's own Codex CLI dropped Chat Completions
  support entirely.
- Use **`openai-chat-completions`** for a server that does not serve `/v1/responses`, which is still
  a great many of them.

Neither is the legacy `/v1/completions` endpoint, which is a third protocol with no tool calling that
meka does not implement.

## API Details

**Endpoint:** `POST {base_url}/responses`
**Auth:** `Authorization: Bearer <api key>`
**Streaming:** SSE, always. `complete` folds the stream internally rather than issuing a separate
non-streaming request.

Request body fields meka sets: `model`, `input`, `instructions` (the system prompt, when non-empty),
`tools`, `tool_choice: auto`, `parallel_tool_calls: false`, `store: false`, `stream: true`,
`reasoning.effort` (only when `effort` is set), and `max_output_tokens` (only when
`max_output_tokens` is set).

What it deliberately does **not** send is `include: ["reasoning.encrypted_content"]` or
`reasoning.summary`. Both are OpenAI extensions: the first round-trips reasoning across stateless
turns, the second asks for the human-readable digest meka renders as a thinking block.
`chatgpt-subscription` sends both because its endpoint is always ChatGPT; here the endpoint is
whatever `base_url` names, meka has no way to know whether either is understood, and an unrecognised
field is a rejected request rather than a degraded one.

The trade-off, stated plainly: against OpenAI itself this backend shows no thinking and carries no
reasoning between a turn's own tool calls. Use `chatgpt-subscription` if you want either. Endpoints
that stream reasoning unprompted (vLLM and Ollama emit `response.reasoning_text.delta` without
being asked) still render their thinking here.
