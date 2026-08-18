# OpenAI Codex Provider

The `openai-codex` provider talks to OpenAI's subscription endpoint using the OAuth tokens issued by ChatGPT login. It's the OpenAI counterpart to [`claude-oauth`](./claude-oauth.md): instead of paying per-token via an API key, you authenticate with your ChatGPT Plus / Pro / Team / Business / Enterprise account and your usage counts against your subscription.

> **Note:** This provider replicates the wire shape that OpenAI's first-party [Codex CLI](https://github.com/openai/codex) sends. It targets `chatgpt.com/backend-api/codex/responses` using the **OpenAI Responses API**, a different protocol than [`openai-api`](./openai-api.md), which uses Chat Completions against `api.openai.com`. The two providers are not interchangeable.

## Configuration

| Setting | Value |
|---------|-------|
| Profile `type` | `openai-codex` |
| Default base URL | `https://chatgpt.com` (request path `/backend-api/codex/responses`) |
| Credential | OAuth bundle stored in the database (acquired via `meka provider add` / `login`) |
| Auth method | OAuth 2.0 Authorization Code with PKCE |
| OAuth issuer | `https://auth.openai.com` |
| Required tier | ChatGPT Plus, Pro, Team, Business, Enterprise, or Edu |

## Initial Setup

```bash
meka provider add chatgpt --type openai-codex --model gpt-5
# A browser opens; sign in to ChatGPT and approve.
# Tokens are saved to ~/.local/share/meka/meka.db (chmod 0600).
```

`meka provider add` binds a local listener on `127.0.0.1:1455` to receive the OAuth callback, matching the redirect URI registered with OpenAI's auth server. If port 1455 is already in use (e.g. you're already running the Codex CLI), free it first.

On a remote or headless machine (SSH, container) the browser runs elsewhere, so the redirect to `http://localhost:1455/...` can't reach meka. In that case, after approving in your browser, copy the full callback URL from the address bar (visible even though the page failed to load) and paste it at the prompt; meka picks the `code` and `state` out of it. The paste prompt runs alongside the local listener, so on a local machine the callback still completes automatically with nothing to paste.

## Config File

`meka provider add` writes this for you (the token bundle stays in the database):

```toml
default_provider = "chatgpt"

[providers.chatgpt]
type = "openai-codex"
model = "gpt-5.6-sol"
effort = "xhigh"   # optional; unset sends none, so OpenAI's default applies
```

The `effort` field maps to the Responses API `reasoning.effort` knob. When unset the `reasoning` block is omitted and OpenAI applies its own default; meka picks no tier and consults no catalog. An explicit value is absolute: sent verbatim, never clamped.

## Supported Models

Whatever your ChatGPT subscription tier exposes: typically `gpt-5`, `gpt-5-codex`, `o3`, `o4-mini`, etc. The model field on the request body is forwarded verbatim; meka doesn't gate which model strings are valid.

## How It Works

Each request:

1. **Auth header set**: `Authorization: Bearer <access_token>`, `ChatGPT-Account-ID: <workspace_id>` (extracted from the JWT id_token at login), `originator: meka_cli`, plus a `User-Agent` identifying meka.
2. **Cookie jar enabled**: `chatgpt.com` is fronted by Cloudflare; bot-clearance cookies (`__cf_bm` etc.) persist across requests automatically.
3. **Body**: standard Responses API JSON: `instructions`, `input` (an array of `message` / `function_call` / `function_call_output` items), `tools`, optional `reasoning.effort`.
4. **Stream**: SSE events: `response.output_text.delta` for text, `response.output_item.added` / `…done` for tool calls, `response.reasoning_text.delta` for thinking, `response.completed` for end-of-turn with token usage.
5. **Token refresh**: when the access token is within 5 minutes of expiry, meka transparently refreshes it against `auth.openai.com/oauth/token` before the next request.

## Limitations

- **Streaming-only**: the Codex endpoint doesn't support non-streaming completions. meka always streams for this provider; `--no-stream` is rejected with an explicit error.
- **Subscription required**: you need a paid ChatGPT plan with Codex enabled. Free-tier accounts can complete the OAuth flow but most models will reject requests at the API layer.
- **Bot detection**: chatgpt.com may serve a Cloudflare challenge if request patterns look automated. meka's reqwest client handles cookie-clearance automatically; if you hit a hard challenge, complete it once in a regular browser to refresh the cookies.
- **Endpoint stability**: this is OpenAI's subscription-internal API; OpenAI doesn't guarantee compatibility for third-party clients. Future Codex versions could add request signing or rotate scopes; meka will need updates if that happens.

## Subscription vs API Key

If you have both a ChatGPT subscription and an OpenAI API key:

- Use **`openai-codex`** for interactive work: it's billed against your subscription's usage cap rather than per-token, so heavy use is cheaper for most personal patterns.
- Use **`openai-api`** for scripted / unattended work: API keys are stable, work with non-OpenAI Chat-Completions-compatible servers (Ollama, vLLM, OpenRouter), and don't depend on the Cloudflare cookie jar.

## Logging Out

`meka provider remove <name>` revokes the OAuth token (best-effort), deletes the stored credential
from the database, and removes the profile from the config file:

```bash
meka provider remove chatgpt
```

To re-authenticate the same profile without removing it (e.g. after a dead refresh token), run
`meka provider login <name>` for a fresh PKCE pair.
