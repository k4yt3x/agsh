# Claude API Provider

The `claude-api` provider talks to the [Claude Messages API](https://docs.anthropic.com/en/api/messages) directly with an `x-api-key` header. Use this when you have a Claude API key (billed per-token). For Claude Code OAuth, see [`claude-oauth`](./claude-oauth.md).

## Configuration

| Setting | Value |
|---------|-------|
| Profile `type` | `claude-api` |
| Default base URL | `https://api.anthropic.com` |
| Credential | API key (`sk-ant-api03-...`) stored in the database |
| Auth method | `x-api-key` header |
| API version | `2023-06-01` |

### Quickest Start

```bash
meka provider add anthropic --type claude-api --model claude-opus-5
```

`meka provider add` prompts for your Claude API key, stores it in the database, and writes the
`[providers.anthropic]` profile. To read the key from a pipe instead of prompting, pass
`--api-key-stdin`.

### Config File

`meka provider add` writes this for you (the key stays in the database, not here):

```toml
default_provider = "anthropic"

[providers.anthropic]
type = "claude-api"
model = "claude-opus-5"
```

### `effort`

meka sends the reasoning-effort control as `output_config.effort` in the request body. Unlike `claude-oauth`, no beta header is needed: the parameter is generally available on the direct Messages API. When `effort` is unset the field is omitted entirely, which is how you ask for Anthropic's own default. See the [`effort`](../configuration/config-file.md#effort) config reference for the levels.

### `thinking`

`adaptive` (the default) sends `thinking: {"type": "adaptive"}` and lets the model set its own budget; `budgeted` sends the older `{"type": "enabled", "budget_tokens": N}` form, taking N from [`[thinking].budget_tokens`](../configuration/config-file.md#thinkingbudget_tokens); `off` sends no thinking field. Pre-4.6 Claude models require `budgeted`.

## Supported Models

Any model available through the Claude Messages API; meka forwards the model string verbatim and doesn't gate which strings are valid. For the current line-up and their retirement dates, see [Anthropic's models overview](https://docs.claude.com/en/docs/about-claude/models/overview) - `meka provider add` suggests `claude-opus-5` for new Claude profiles.

## Custom Base URL

To use a Claude-API-compatible proxy or gateway:

```bash
meka --provider claude-api --model claude-opus-5 --base-url https://my-proxy.example.com
```

A trailing `/v1` is dropped, since meka appends it per request: publish `https://gateway.example.com/anthropic` or `https://gateway.example.com/anthropic/v1`, either works.

### Anthropic-compatible endpoints

The model behind the endpoint doesn't have to be Claude. Ollama, LM Studio and similar runtimes serve local weights over `POST /v1/messages`, and `claude-api` reaches them with a placeholder key:

```bash
meka provider add local --type claude-api \
    --model 'hf.co/bartowski/Qwen3.8-27B-GGUF:Q8_0' \
    --base-url http://127.0.0.1:11434
```

Nothing in the request is tuned to Claude unless you ask for it. `effort` is omitted when unset, so a backend with no reasoning tiers is never handed one, and `thinking` is whatever the profile says rather than something inferred from the model name - set `budgeted` if your endpoint only implements the older encoding, or `off` if it implements neither.

The one setting worth stating is the context window. meka never probes for it, so an unset profile budgets against the 1M default; on a smaller model that means compaction only fires once the backend itself rejects the request:

```toml
[providers.local]
context_window = 262144
thinking = "budgeted"   # only if the endpoint rejects the adaptive form
```

## API Details

**Endpoint:** `POST {base_url}/v1/messages`

**Headers:**
- `x-api-key: <api_key>`
- `anthropic-version: 2023-06-01`
- `content-type: application/json`

**System prompt:** Sent as a top-level `system` string.

**Tool format:** Tools are defined with `input_schema`:

```json
{
  "name": "read_file",
  "description": "Read the contents of a file at the given path.",
  "input_schema": { "type": "object", "properties": { ... } }
}
```

**Streaming:** Server-Sent Events with named event types (`message_start`, `content_block_start`, `content_block_delta`, `content_block_stop`, `message_delta`, `message_stop`, `ping`).
