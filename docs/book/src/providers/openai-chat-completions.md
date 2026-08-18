# OpenAI Chat Completions

The **Chat Completions API** (`POST {base_url}/chat/completions`) with an API key. Works against OpenAI and any endpoint implementing that format: Ollama, vLLM, LM Studio, OpenRouter, Synthetic, LiteLLM.

This is *not* the legacy `/v1/completions` endpoint, which is a different protocol: a bare `prompt` string in, `choices[].text` out, no tool calling. Several of those same servers also expose it; meka does not implement it.

For the same key against OpenAI's newer protocol, see [`openai-responses`](./openai-responses.md).

## Configuration

| Setting | Value |
|---------|-------|
| Profile `type` | `openai-chat-completions` |
| Default base URL | `https://api.openai.com/v1` |
| Credential | API key (`sk-...`) stored in the database |
| Auth method | Bearer token (`Authorization: Bearer <key>`) |

### Quickest Start

```bash
meka provider add openai --type openai-chat-completions --model gpt-4o
```

`meka provider add` prompts for your OpenAI API key, stores it in the database, and writes the
`[providers.openai]` profile. To read the key from a pipe instead of prompting, pass
`--api-key-stdin`.

### Config File

`meka provider add` writes this for you (the key stays in the database, not here):

```toml
default_provider = "openai"

[providers.openai]
type = "openai-chat-completions"
model = "gpt-4o"
```

## Supported Models

Any model available through the OpenAI Chat Completions API (or compatible endpoint) that supports tool calling:

- `gpt-4o`, `gpt-4o-mini`
- `gpt-4-turbo`
- `o1`, `o3-mini`
- Third-party models via compatible APIs

## Custom Base URL

To use an OpenAI-compatible endpoint, set the profile's `base_url`. Add it when creating the profile:

```bash
# Ollama (no real key; pipe a placeholder)
printf 'unused' | meka provider add ollama --type openai-chat-completions --model llama3 \
    --base-url http://localhost:11434/v1 --api-key-stdin

# OpenRouter
meka provider add openrouter --type openai-chat-completions --model anthropic/claude-sonnet-4.6 \
    --base-url https://openrouter.ai/api/v1
```

The resulting profile (the key, if any, lives in the database):

```toml
[providers.ollama]
type = "openai-chat-completions"
model = "llama3"
base_url = "http://localhost:11434/v1"
```

`--base-url` is also available as a per-run flag to override the profile's value for one invocation.

## API Details

**Endpoint:** `POST {base_url}/chat/completions`

**Tool format:** Tools are sent as function definitions:

```json
{
  "type": "function",
  "function": {
    "name": "read_file",
    "description": "Read the contents of a file at the given path.",
    "parameters": { "type": "object", "properties": { ... } }
  }
}
```

**Tool results:** Sent back as messages with `role: "tool"` and the corresponding `tool_call_id`.

**Streaming:** Uses Server-Sent Events (SSE) with `data: {...}` lines. The stream ends with `data: [DONE]`.
