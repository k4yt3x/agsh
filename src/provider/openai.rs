//! OpenAI-flavoured providers.
//!
//! Three live here, across two protocols. OpenAI serves both, which is why the backend names say
//! which one rather than naming the vendor twice:
//!
//! - [`chat_completions`]: Chat Completions, `POST {base}/chat/completions`, against
//!   `api.openai.com/v1` or any endpoint implementing that format (Ollama, vLLM, OpenRouter,
//!   Synthetic, …). Bearer-token auth. Note this is *not* the legacy `/v1/completions`, a different
//!   protocol several of those same servers also expose.
//! - [`responses`]: the Responses API, `POST {base}/responses`, likewise against OpenAI or any
//!   endpoint serving it (Ollama v0.13.3+, vLLM, LM Studio, OpenRouter). Bearer-token auth.
//! - [`subscription`]: the Responses API against `chatgpt.com/backend-api/codex`, authenticated by
//!   ChatGPT subscription OAuth (Plus / Pro / Team / Business / Enterprise) and shaped like
//!   OpenAI's own Codex CLI.
//!
//! The two Responses backends share the wire format through [`responses_wire`]; Chat Completions
//! is a different protocol and shares nothing but [`data_url`].

pub mod chat_completions;
pub mod responses;
pub(crate) mod responses_wire;
pub mod subscription;

pub use chat_completions::OpenAiChatCompletionsProvider;
pub use responses::OpenAiResponsesProvider;
pub use subscription::ChatGptSubscriptionProvider;

/// A `data:` URL for an image, the one piece of image wire-format both protocols share (Chat
/// Completions `image_url.url` and the Responses API `input_image.image_url`).
fn data_url(source: &crate::provider::ImageSource) -> String {
    format!("data:{};base64,{}", source.media_type, source.data)
}
