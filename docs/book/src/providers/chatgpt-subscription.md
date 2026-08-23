# ChatGPT subscription

The **Responses API** billed to a ChatGPT subscription, at `chatgpt.com/backend-api/codex/responses`, using the OAuth tokens issued by ChatGPT login and mirroring the first-party Codex CLI's request shape. The OpenAI counterpart to [`claude-subscription`](./claude-subscription.md): instead of paying per token via an API key, you authenticate with your ChatGPT Plus / Pro / Team / Business / Enterprise account and usage counts against your subscription.

For the same protocol with an API key, against OpenAI or any server that serves it, see [`openai-responses`](./openai-responses.md).

> **Note:** This provider replicates the wire shape that OpenAI's first-party [Codex CLI](https://github.com/openai/codex) sends. It targets `chatgpt.com/backend-api/codex/responses` using the **OpenAI Responses API**, a different protocol than [`openai-chat-completions`](./openai-chat-completions.md), which uses Chat Completions against `api.openai.com`. The two providers are not interchangeable.

## Configuration

| Setting | Value |
|---------|-------|
| Profile `type` | `chatgpt-subscription` |
| Default base URL | `https://chatgpt.com` (request path `/backend-api/codex/responses`) |
| Credential | OAuth bundle stored in the database (acquired via `meka provider add` / `login`) |
| Auth method | OAuth 2.0 Authorization Code with PKCE |
| OAuth issuer | `https://auth.openai.com` |
| Required tier | ChatGPT Plus, Pro, Team, Business, Enterprise, or Edu |

## Initial Setup

```bash
meka provider add chatgpt --type chatgpt-subscription --model gpt-5.6-sol
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
type = "chatgpt-subscription"
model = "gpt-5.6-sol"
effort = "xhigh"   # optional; unset sends none, so OpenAI's default applies
```

The `effort` field maps to the Responses API `reasoning.effort` knob. When unset the `reasoning` block is omitted and OpenAI applies its own default; meka picks no tier and consults no catalog. An explicit value is absolute: sent verbatim, never clamped.

## Supported Models

Whatever your ChatGPT subscription tier exposes. For the current line-up, see [OpenAI's models overview](https://platform.openai.com/docs/models) - `meka provider add` suggests `gpt-5.6-sol` for new OpenAI profiles. The model field on the request body is forwarded verbatim; meka doesn't gate which model strings are valid.

## How It Works

Each request:

1. **Auth header set**: `Authorization: Bearer <access_token>`, `ChatGPT-Account-ID: <workspace_id>` (extracted from the JWT id_token at login), `originator: meka_cli`, plus a `User-Agent` identifying meka.
2. **Cookie jar enabled**: `chatgpt.com` is fronted by Cloudflare; bot-clearance cookies (`__cf_bm` etc.) persist across requests automatically.
3. **Body**: standard Responses API JSON: `instructions`, `input` (an array of `message` / `reasoning` / `function_call` / `function_call_output` items), `tools`, optional `reasoning.effort`, plus the two reasoning parameters Codex also sends: `reasoning.summary: "auto"` and `include: ["reasoning.encrypted_content"]`. Both are sent on every request, whether or not `effort` is configured.
4. **Stream**: SSE events: `response.output_text.delta` for text, `response.output_item.added` / `…done` for tool calls, `response.reasoning_summary_text.delta` (and `response.reasoning_text.delta`) for thinking, `response.reasoning_summary_part.added` for the break between summary sections, `response.completed` for end-of-turn with token usage.

### Reasoning across turns

Requests are stateless (`store: false`), so the reasoning a model produced is only available to the next request if meka sends it back. It does: each reasoning item is recorded with its `rs_…` id and its `encrypted_content`, and replayed verbatim as a `reasoning` input item immediately before the output it produced. This is what lets a multi-step tool-calling turn keep one chain of thought instead of restarting it at every call, and it mirrors what the first-party Codex client does.

The encrypted content is opaque: meka cannot read it, only replay it. It is stored under a shape that records which provider it came from, so a session recorded here and resumed against Claude does not hand Claude an OpenAI blob (nor the reverse); a block from the wrong provider is simply not replayed. The summary is the readable part, and what the REPL shows as a thinking block (see [`[thinking]`](../configuration/config-file.md) for `show_content`).

A session recorded by 0.41 holds its thinking blocks under a shape that names no provider, and meka does not reshape them when it opens a session. The [one-shot upgrade script](../getting-started/upgrading.md) does it, in a pass over the database you can watch finish, for the same reason the [memory import](../usage/memory.md#coming-from-a-file-backed-store) is a script: a migration that runs on every start is one nobody can see fail. Until it runs, such a block keeps its readable summary and loses its encrypted half, so that reasoning is not replayed.
5. **Token refresh**: when the access token is within 5 minutes of expiry, meka transparently refreshes it against `auth.openai.com/oauth/token` before the next request.

## Limitations

- **Streaming-only**: the Codex endpoint has no non-streaming shape, so meka always streams here and folds the stream internally to satisfy a non-streaming completion. `--no-stream` is accepted and behaves normally; it changes what the terminal renders, not what goes on the wire.
- **Subscription required**: you need a paid ChatGPT plan with Codex enabled. Free-tier accounts can complete the OAuth flow but most models will reject requests at the API layer.
- **Bot detection**: chatgpt.com may serve a Cloudflare challenge if request patterns look automated. meka's reqwest client handles cookie-clearance automatically; if you hit a hard challenge, complete it once in a regular browser to refresh the cookies.
- **Endpoint stability**: this is OpenAI's subscription-internal API; OpenAI doesn't guarantee compatibility for third-party clients. Future Codex versions could add request signing or rotate scopes; meka will need updates if that happens.

## Subscription vs API Key

If you have both a ChatGPT subscription and an OpenAI API key:

- Use **`chatgpt-subscription`** for interactive work: it's billed against your subscription's usage cap rather than per-token, so heavy use is cheaper for most personal patterns.
- Use **`openai-responses`** for scripted / unattended work: it is the same protocol as this backend with a plain API key, so keys are stable, nothing depends on the Cloudflare cookie jar, and it also reaches Ollama, vLLM, LM Studio and OpenRouter. Fall back to **`openai-chat-completions`** for a server that does not serve `/v1/responses`.

## Logging Out

`meka provider remove <name>` revokes the OAuth token (best-effort), deletes the stored credential
from the database, and removes the profile from the config file:

```bash
meka provider remove chatgpt
```

To re-authenticate the same profile without removing it (e.g. after a dead refresh token), run
`meka provider login <name>` for a fresh PKCE pair.
