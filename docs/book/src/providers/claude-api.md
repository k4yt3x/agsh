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
meka provider add anthropic --type claude-api --model claude-opus-4-6
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
model = "claude-opus-4-6"
```

### `effort`

meka sends the reasoning-effort control as `output_config.effort` in the request body. Unlike `claude-oauth`, no beta header is needed: the parameter is generally available on the direct Messages API. When `effort` is unset, meka picks a model-aware default (`xhigh` where supported, else `high`, omitted on models with no effort knob). See the [`effort`](../configuration/config-file.md#effort) config reference for the full per-model matrix and the supported levels.

## Supported Models

Any model available through the Claude Messages API. Current line-up (per [Anthropic's models overview](https://docs.claude.com/en/docs/about-claude/models/overview)):

| Family | Alias | Notes |
|--------|-------|-------|
| Opus 4.7 | `claude-opus-4-7` | Latest Opus; most capable, adaptive thinking |
| Sonnet 4.6 | `claude-sonnet-4-6` | Latest Sonnet, speed + intelligence balance |
| Haiku 4.5 | `claude-haiku-4-5` | Latest Haiku, fastest |

Older but still available: `claude-opus-4-6`, `claude-sonnet-4-5`, `claude-opus-4-5`, `claude-opus-4-1`. Deprecated and retiring 2026-06-15: `claude-opus-4-20250514`, `claude-sonnet-4-20250514`.

## Custom Base URL

To use a Claude-API-compatible proxy or gateway:

```bash
meka --provider claude-api --model claude-opus-4-6 --base-url https://my-proxy.example.com
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
