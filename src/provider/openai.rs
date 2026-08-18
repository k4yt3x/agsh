//! OpenAI-flavoured providers.
//!
//! Two siblings live here, intentionally not sharing protocol code:
//!
//! - [`api`]: Chat Completions against `api.openai.com/v1` or any OpenAI-compatible endpoint
//!   (Ollama, vLLM, OpenRouter, …). Bearer-token auth via `OPENAI_API_KEY`.
//! - [`codex`]: OpenAI Responses API against `chatgpt.com/backend-api/codex`, authenticated by
//!   ChatGPT subscription OAuth (Plus / Pro / Team / Business / Enterprise). Mirrors how OpenAI's
//!   own first-party Codex CLI talks to the subscription endpoint. The protocol differs from
//!   `api`'s Chat Completions, so they don't share request/response code.

pub mod api;
pub mod codex;

pub use api::OpenAiProvider;
pub use codex::OpenAiCodexProvider;

/// A `data:` URL for an image, the one piece of image wire-format both sub-providers share (Chat
/// Completions `image_url.url` and the Responses API `input_image.image_url`).
fn data_url(source: &crate::provider::ImageSource) -> String {
    format!("data:{};base64,{}", source.media_type, source.data)
}
