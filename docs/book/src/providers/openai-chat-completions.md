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
meka provider add openai --type openai-chat-completions --model gpt-5.6-sol
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
model = "gpt-5.6-sol"
```

## Supported Models

Any model reachable over the Chat Completions API that supports tool calling. For OpenAI's current line-up, see [OpenAI's models overview](https://platform.openai.com/docs/models) - `meka provider add` suggests `gpt-5.6-sol` for new OpenAI profiles. Against a compatible endpoint the valid names are that server's: whatever Ollama, vLLM, LM Studio or OpenRouter serves. meka forwards the model string verbatim and doesn't gate which strings are valid.

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

Change it later with `meka provider set <name> base_url <url>`.

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
