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

/// Lowercase and drop a leading `vendor/` path segment (e.g. OpenRouter's `openai/gpt-5.6`) so the
/// family/version checks below see the bare model slug.
fn normalized_model(model: &str) -> String {
    model
        .rsplit('/')
        .next()
        .unwrap_or(model)
        .to_ascii_lowercase()
}

/// Parse the `(major, minor)` version right after a `gpt-` prefix: `gpt-5.6-sol` -> `(5, 6)`,
/// `gpt-5` -> `(5, 0)`, `gpt-4.1` -> `(4, 1)`. `None` when there's no `gpt-<digit>` (e.g. an
/// o-series or non-OpenAI name).
fn gpt_version(model: &str) -> Option<(u32, u32)> {
    let rest = &model[model.find("gpt-")? + 4..];
    let major_digits: String = rest
        .chars()
        .take_while(|byte| byte.is_ascii_digit())
        .collect();
    let major = major_digits.parse().ok()?;
    let minor = rest
        .strip_prefix(&major_digits)
        .and_then(|tail| tail.strip_prefix('.'))
        .map(|tail| {
            tail.chars()
                .take_while(|byte| byte.is_ascii_digit())
                .collect::<String>()
        })
        .and_then(|digits| digits.parse().ok())
        .unwrap_or(0);
    Some((major, minor))
}

/// Whether an OpenAI model exposes the `reasoning.effort` knob. Recognized reasoning families only,
/// so unknown names (local models served through `openai-api`: Ollama, vLLM, OpenRouter
/// passthrough) get the field omitted rather than a possibly-rejected `reasoning` block. Future
/// OpenAI reasoning models follow the `gpt-<major>=5+` / `o<n>` naming and are covered
/// optimistically.
pub(crate) fn model_supports_effort(model: &str) -> bool {
    let model = normalized_model(model);
    // Non-reasoning members of otherwise-reasoning families: the `-chat` GPT models
    // (`gpt-5-chat-latest`, ...) are the plain ChatGPT model, and o1's `mini`/`preview` spinoffs
    // predate reasoning-effort support. All three reject the `reasoning.effort` knob.
    if model.contains("-chat") || model.starts_with("o1-mini") || model.starts_with("o1-preview") {
        return false;
    }
    if let Some((major, _)) = gpt_version(&model) {
        return major >= 5;
    }
    model.starts_with("o1")
        || model.starts_with("o3")
        || model.starts_with("o4")
        || model.starts_with("o5")
        || model.contains("codex-mini")
}

/// Whether an OpenAI model supports the `xhigh` effort tier. `xhigh` arrived with
/// `gpt-5.1-codex-max` and is standard from `gpt-5.2` onward; earlier reasoning models (o-series,
/// `gpt-5`, `gpt-5.1`, `gpt-5-codex`/`mini`/`nano`/`pro`) top out at `high`.
pub(crate) fn model_supports_xhigh(model: &str) -> bool {
    if !model_supports_effort(model) {
        return false;
    }
    let model = normalized_model(model);
    if model.contains("codex-max") {
        return true;
    }
    match gpt_version(&model) {
        Some((major, _)) if major > 5 => true,
        Some((5, minor)) => minor >= 2,
        _ => false,
    }
}

/// Resolve the `reasoning.effort` value for `model` given the profile's explicit override (`None` =
/// unset). A `None` return means the caller omits the `reasoning` block. See
/// [`crate::provider::resolve_effort_level`] for the shared policy.
pub(crate) fn resolve_reasoning_effort(configured: Option<&str>, model: &str) -> Option<String> {
    crate::provider::resolve_effort_level(
        configured,
        model_supports_effort(model),
        model_supports_xhigh(model),
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_supports_effort() {
        for reasoning in [
            "gpt-5.6-sol",
            "gpt-5.2",
            "gpt-5.1",
            "gpt-5",
            "gpt-5-codex",
            "gpt-5-pro",
            "o1",
            "o3",
            "o3-mini",
            "o4-mini",
            "codex-mini-latest",
            "openai/gpt-5.6-terra",
        ] {
            assert!(
                model_supports_effort(reasoning),
                "{reasoning} should support effort"
            );
        }
        for non_reasoning in [
            "o1-mini",
            "o1-preview",
            "gpt-4o",
            "gpt-4.1",
            "gpt-3.5-turbo",
            // The non-reasoning ChatGPT models share the gpt-5 major but reject reasoning.effort.
            "gpt-5-chat-latest",
            "gpt-5-chat",
            "chatgpt-4o-latest",
            "llama3.1",
            "qwen2.5-coder",
            "mistral-large",
            "",
        ] {
            assert!(
                !model_supports_effort(non_reasoning),
                "{non_reasoning} should omit effort"
            );
        }
    }

    #[test]
    fn test_model_supports_xhigh() {
        for yes in [
            "gpt-5.6-sol",
            "gpt-5.5",
            "gpt-5.2",
            "gpt-5.2-codex",
            "gpt-5.1-codex-max",
            "openai/gpt-5.4",
        ] {
            assert!(model_supports_xhigh(yes), "{yes} should support xhigh");
        }
        for no in [
            "gpt-5.1",
            "gpt-5",
            "gpt-5-codex",
            "gpt-5-chat-latest",
            "o3",
            "o4-mini",
            "gpt-4o",
            "llama3.1",
        ] {
            assert!(!model_supports_xhigh(no), "{no} should not support xhigh");
        }
    }

    #[test]
    fn test_resolve_reasoning_effort() {
        // Unset: strongest supported tier.
        assert_eq!(
            resolve_reasoning_effort(None, "gpt-5.6-sol").as_deref(),
            Some("xhigh")
        );
        assert_eq!(
            resolve_reasoning_effort(None, "gpt-5.1").as_deref(),
            Some("high")
        );
        assert_eq!(
            resolve_reasoning_effort(None, "o3").as_deref(),
            Some("high")
        );
        // Explicit values are absolute: passed through verbatim, never clamped.
        assert_eq!(
            resolve_reasoning_effort(Some("xhigh"), "gpt-5.1").as_deref(),
            Some("xhigh")
        );
        assert_eq!(
            resolve_reasoning_effort(Some("low"), "gpt-5.6-sol").as_deref(),
            Some("low")
        );
        assert_eq!(
            resolve_reasoning_effort(Some("max"), "gpt-5.6-sol").as_deref(),
            Some("max")
        );
        // ...even on a model the default would omit the field for (unrecognized / non-reasoning).
        assert_eq!(
            resolve_reasoning_effort(Some("high"), "o1-mini").as_deref(),
            Some("high")
        );
        // Unset on an unrecognized / non-reasoning model omits the field.
        assert_eq!(resolve_reasoning_effort(None, "llama3.1"), None);
    }
}
