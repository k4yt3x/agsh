//! Resolving a model's context window, best-effort and self-healing.
//!
//! Precedence (highest first): an explicit config value; the built-in table for recognized models
//! ([`crate::config::context_window_for_model`]); a fresh DB cache entry; a live probe of the
//! provider's models API ([`Provider::fetch_model_info`]); then a 128k floor. The window only
//! drives compaction timing / keep-budget / the `/status` gauge, and overflow-recovery is the
//! runtime backstop, so every step here is fail-soft and the whole thing never blocks startup for
//! more than `PROBE_TIMEOUT`. See `docs/book/src/configuration/config-file.md`.

use std::{sync::Arc, time::Duration};

use crate::{config::context_window_for_model, provider::Provider, session::TokenStore};

/// Floor used when nothing else resolves (and for the negative-cache marker).
const DEFAULT_CONTEXT_WINDOW: u64 = 128_000;
/// Lifetime of an API-sourced cache entry. Windows rarely change, so cache aggressively.
const API_TTL: Duration = Duration::from_secs(30 * 24 * 60 * 60);
/// Lifetime of the negative marker written when the API can't resolve an unknown model. Short so a
/// recovered API or a table fix supersedes it soon, without re-probing (and eating the timeout) on
/// every launch in the meantime.
const NEGATIVE_TTL: Duration = Duration::from_secs(6 * 60 * 60);
/// Upper bound on one models-API probe so a slow or hung endpoint never stalls agent assembly.
const PROBE_TIMEOUT: Duration = Duration::from_secs(5);

/// Provenance of a cached window, which selects its TTL. Persisted as the `source` TEXT column; the
/// enum keeps the string values and their TTLs in one place instead of scattering `== "api"`
/// checks.
#[derive(Clone, Copy)]
enum CacheSource {
    /// Resolved from the provider's models API. Cached long; windows rarely change.
    Api,
    /// The 128k floor, written when the API couldn't resolve an unrecognized model. Short-lived so
    /// a recovered API or a table fix supersedes it soon rather than re-probing every launch.
    Resolver,
}

impl CacheSource {
    fn as_str(self) -> &'static str {
        match self {
            CacheSource::Api => "api",
            CacheSource::Resolver => "resolver",
        }
    }

    /// TTL for a stored `source` string. An unrecognized value is treated as the short `Resolver`
    /// TTL (conservative: re-resolve sooner).
    fn ttl_for(source: &str) -> Duration {
        if source == CacheSource::Api.as_str() {
            API_TTL
        } else {
            NEGATIVE_TTL
        }
    }
}

/// Resolve the context window for the active model, probing the provider's models API for a model
/// the built-in table doesn't recognize and caching the result keyed by `(profile, model)`.
/// `config_window` is the resolved `[providers.<name>].context_window` / `[session].context_window`
/// override; `profile` is the active profile name (the cache key) and `model` the active model.
pub async fn resolve_context_window(
    config_window: Option<u64>,
    provider: &Arc<dyn Provider>,
    token_store: &TokenStore,
    profile: Option<&str>,
    model: Option<&str>,
) -> u64 {
    let Some(model) = model else {
        return config_window.unwrap_or(DEFAULT_CONTEXT_WINDOW);
    };
    // Steps 1-3: override / table / cache, no network.
    if let Some(window) = resolve_offline(config_window, token_store, profile, model).await {
        return window;
    }
    // Step 4: probe the provider's models API (bounded, fail-soft). Warn on failure so a user
    // seeing early compaction can tell the window fell back rather than being detected.
    match tokio::time::timeout(PROBE_TIMEOUT, provider.fetch_model_info()).await {
        Ok(Ok(Some(info))) => {
            if let Some(window) = info.context_window {
                save(token_store, profile, model, window, CacheSource::Api).await;
                return window;
            }
        }
        Ok(Ok(None)) => {}
        Ok(Err(error)) => {
            tracing::warn!(
                "model context-window probe failed, using fallback: {}",
                error
            )
        }
        Err(_elapsed) => tracing::warn!(
            "model context-window probe timed out after {:?}, using fallback",
            PROBE_TIMEOUT
        ),
    }
    // Step 5: floor. Leave a short-lived negative marker so we don't re-probe every launch.
    save(
        token_store,
        profile,
        model,
        DEFAULT_CONTEXT_WINDOW,
        CacheSource::Resolver,
    )
    .await;
    DEFAULT_CONTEXT_WINDOW
}

/// Resolve without a network probe: override / table / cache / floor. Used where no provider handle
/// is available (the REPL prompt gauge). The agent's own [`resolve_context_window`] runs alongside
/// and populates the cache, so an unknown model's gauge converges to the accurate value by the next
/// launch (and matches immediately for recognized models).
pub async fn resolve_context_window_cached(
    config_window: Option<u64>,
    token_store: &TokenStore,
    profile: Option<&str>,
    model: Option<&str>,
) -> u64 {
    let Some(model) = model else {
        return config_window.unwrap_or(DEFAULT_CONTEXT_WINDOW);
    };
    resolve_offline(config_window, token_store, profile, model)
        .await
        .unwrap_or(DEFAULT_CONTEXT_WINDOW)
}

/// The non-probing prefix of the chain: config override, then the built-in table, then a fresh DB
/// cache entry (TTL by source). `None` only when `model` is unrecognized and has no fresh cache
/// entry, i.e. the point at which the caller would probe or floor.
async fn resolve_offline(
    config_window: Option<u64>,
    token_store: &TokenStore,
    profile: Option<&str>,
    model: &str,
) -> Option<u64> {
    if let Some(window) = config_window {
        return Some(window);
    }
    if let Some(window) = context_window_for_model(model) {
        return Some(window);
    }
    if let Some(profile) = profile {
        match token_store.load_model_context(profile, model).await {
            Ok(Some((window, source, age))) => {
                if age < CacheSource::ttl_for(&source).as_secs() as i64 {
                    return Some(window);
                }
            }
            Ok(None) => {}
            Err(error) => tracing::debug!("model-context cache read failed: {}", error),
        }
    }
    None
}

/// Persist a resolved window to the cache, best-effort. No-op without a profile to key on.
async fn save(
    token_store: &TokenStore,
    profile: Option<&str>,
    model: &str,
    window: u64,
    source: CacheSource,
) {
    let Some(profile) = profile else { return };
    if let Err(error) = token_store
        .save_model_context(profile, model, window, source.as_str())
        .await
    {
        tracing::debug!("model-context cache write failed: {}", error);
    }
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;
    use crate::{
        provider::{ModelInfo, mock::MockProvider},
        session::SessionManager,
    };

    async fn token_store() -> TokenStore {
        SessionManager::open(Some(Path::new(":memory:")))
            .await
            .expect("in-memory db")
            .token_store()
    }

    fn provider_with(context_window: Option<u64>) -> Arc<dyn Provider> {
        Arc::new(MockProvider::with_model_info(ModelInfo {
            context_window,
            max_output_tokens: None,
        }))
    }

    #[tokio::test]
    async fn config_override_wins_unprobed_and_uncached() {
        let store = token_store().await;
        let provider = provider_with(Some(999)); // would answer 999 if reached
        let window = resolve_context_window(
            Some(500_000),
            &provider,
            &store,
            Some("p"),
            Some("unknown-x"),
        )
        .await;
        assert_eq!(window, 500_000);
        assert!(
            store
                .load_model_context("p", "unknown-x")
                .await
                .unwrap()
                .is_none()
        );
    }

    #[tokio::test]
    async fn recognized_model_uses_table_not_provider() {
        let store = token_store().await;
        let provider = provider_with(Some(42)); // bogus; the table must win for a known model
        let window =
            resolve_context_window(None, &provider, &store, Some("p"), Some("gpt-5.6-sol")).await;
        assert_eq!(window, 1_050_000);
        assert!(
            store
                .load_model_context("p", "gpt-5.6-sol")
                .await
                .unwrap()
                .is_none()
        );
    }

    #[tokio::test]
    async fn api_probe_resolves_then_cache_serves() {
        let store = token_store().await;
        let provider = provider_with(Some(321_000));
        let window =
            resolve_context_window(None, &provider, &store, Some("p"), Some("unknown-y")).await;
        assert_eq!(window, 321_000);
        let (cached, source, _) = store
            .load_model_context("p", "unknown-y")
            .await
            .unwrap()
            .expect("api result cached");
        assert_eq!(cached, 321_000);
        assert_eq!(source, "api");
        // A second resolve serves the cache even if the provider would now answer differently.
        let provider2 = provider_with(Some(1));
        let window2 =
            resolve_context_window(None, &provider2, &store, Some("p"), Some("unknown-y")).await;
        assert_eq!(window2, 321_000);
    }

    #[tokio::test]
    async fn unknown_without_api_floors_and_negative_caches() {
        let store = token_store().await;
        let provider = provider_with(None); // provider reports nothing
        let window =
            resolve_context_window(None, &provider, &store, Some("p"), Some("unknown-z")).await;
        assert_eq!(window, DEFAULT_CONTEXT_WINDOW);
        let (cached, source, _) = store
            .load_model_context("p", "unknown-z")
            .await
            .unwrap()
            .expect("negative marker cached");
        assert_eq!(cached, DEFAULT_CONTEXT_WINDOW);
        assert_eq!(source, "resolver");
    }
}
