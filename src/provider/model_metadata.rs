//! Resolving a model's metadata (context window + reasoning effort), best-effort and self-healing.
//!
//! One post-build pass
//! ([`resolve_model_metadata`](crate::provider::model_metadata::resolve_model_metadata))
//! fetches the provider's `/models` catalog at most once (cached per `(profile, model)`) and shares
//! that single probe and cache across both attributes. Each is then resolved by attribute-specific
//! logic that follows the same precedence shape - `override > authoritative source > fallback >
//! floor` - but the authoritative source differs by attribute:
//!
//! - **context window**: the built-in table ([`crate::provider::context_window_for_model`]) is
//!   authoritative for recognized models (it encodes each model's real window as meka's request
//!   receives it, so it reflects the window the request truly gets); the live probe fills only
//!   models the table doesn't recognize. Precedence: config override > table > cached probe > live
//!   probe > 128k floor.
//! - **effort**: the provider catalog is authoritative when it reports levels (Codex); otherwise
//!   the provider's name predicates decide. Applied to the provider via
//!   [`crate::provider::Provider::refine_effort`].
//!
//! The window drives compaction timing / keep-budget / the `/status` gauge, and overflow-recovery
//! is the runtime backstop, so every step here is fail-soft and the whole pass never blocks startup
//! for more than `PROBE_TIMEOUT`. See `docs/book/src/configuration/config-file.md`.

use std::{sync::Arc, time::Duration};

use crate::{
    provider::{ModelInfo, Provider, context_window_for_model},
    session::TokenStore,
};

/// Floor reported to the caller when nothing else resolves the context window. (The negative-cache
/// marker itself stores no window - an all-`None` [`ModelInfo`] - so a later table/config value can
/// still win; this floor only applies when even that yields nothing.)
const DEFAULT_CONTEXT_WINDOW: u64 = 128_000;
/// Lifetime of an API-sourced cache entry. Metadata rarely changes, so cache aggressively.
const API_TTL: Duration = Duration::from_secs(30 * 24 * 60 * 60);
/// Lifetime of the negative marker written when the API can't resolve a model. Short so a recovered
/// API or a table fix supersedes it soon, without re-probing (and eating the timeout) on every
/// launch in the meantime.
const NEGATIVE_TTL: Duration = Duration::from_secs(6 * 60 * 60);
/// Upper bound on one models-API probe so a slow or hung endpoint never stalls agent assembly.
const PROBE_TIMEOUT: Duration = Duration::from_secs(5);

/// The resolved model metadata handed back to the caller. Effort is not returned here: it is
/// applied directly to the provider (which consumes it at wire time) via
/// [`crate::provider::Provider::refine_effort`] inside [`resolve_model_metadata`].
pub struct ResolvedModel {
    /// Effective context window in tokens (never zero).
    pub context_window: u64,
}

/// Provenance of a cached metadata row, which selects its TTL. Persisted as the `source` TEXT
/// column; the enum keeps the string values and their TTLs in one place instead of scattering
/// `== "api"` checks.
#[derive(Clone, Copy)]
enum CacheSource {
    /// Resolved from the provider's models API. Cached long; metadata rarely changes.
    Api,
    /// The negative marker written when the API couldn't resolve a model. Short-lived so a
    /// recovered API or a table fix supersedes it soon rather than re-probing every launch.
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

/// Resolve the active model's metadata: derive the context window and apply the reasoning-effort
/// default to `provider`. Probes the provider's models API at most once for a model the built-in
/// table doesn't recognize, or for a provider that still needs its effort catalog
/// ([`Provider::needs_effort_catalog`]) even when the window is table-known; caches the whole
/// [`ModelInfo`] keyed by `(profile, model)`. `config_window` is the resolved
/// `[providers.<name>].context_window` / `[session].context_window` override; `profile` is the
/// active profile name (the cache key) and `model` the active model.
pub async fn resolve_model_metadata(
    config_window: Option<u64>,
    provider: &Arc<dyn Provider>,
    token_store: &TokenStore,
    profile: Option<&str>,
    model: Option<&str>,
) -> ResolvedModel {
    let Some(model) = model else {
        // No model name to key the cache / probe on. Effort stays at the provider's construction
        // default (name predicate); the window falls back to the config override or the floor.
        provider.refine_effort(None);
        return ResolvedModel {
            context_window: config_window.unwrap_or(DEFAULT_CONTEXT_WINDOW),
        };
    };

    // A fresh cached ModelInfo, if any (no network). `table_window` is the built-in table's answer
    // for this model (authoritative for recognized models), reused for both the probe decision and
    // the final window.
    let cached = load_cached(token_store, profile, model).await;
    let table_window = context_window_for_model(model);

    // The window as far as the offline sources can tell: config override > built-in table > cache.
    let window_offline = config_window
        .or(table_window)
        .or_else(|| cached.as_ref().and_then(|info| info.context_window));

    // Probe when there's no fresh cache AND either the window is still unknown or this provider
    // still needs its effort catalog (a table-known Codex model with no pinned effort, so it gets
    // catalog-accurate effort). With effort pinned, `needs_effort_catalog` is false, so a
    // table-known window skips the probe entirely. A fresh cache already holds the last probe.
    let need_probe =
        cached.is_none() && (window_offline.is_none() || provider.needs_effort_catalog());

    let fetched = if need_probe {
        probe_and_cache(provider, token_store, profile, model).await
    } else {
        cached
    };

    // Effort: hand the (cached or freshly probed) catalog to the provider, which resolves
    // catalog-first-else-name-predicate and stores the result for the wire path.
    provider.refine_effort(fetched.as_ref());

    // Window: table stays authoritative for recognized models; the probe result fills the rest.
    let context_window = config_window
        .or(table_window)
        .or_else(|| fetched.as_ref().and_then(|info| info.context_window))
        .unwrap_or(DEFAULT_CONTEXT_WINDOW);

    ResolvedModel { context_window }
}

/// Probe the provider's models API (bounded, fail-soft) and cache the outcome. On success caches
/// the full [`ModelInfo`] (`Api` source). On a miss / error / timeout, writes a short-lived
/// negative marker so we don't re-probe every launch, and returns `None`.
async fn probe_and_cache(
    provider: &Arc<dyn Provider>,
    token_store: &TokenStore,
    profile: Option<&str>,
    model: &str,
) -> Option<ModelInfo> {
    match tokio::time::timeout(PROBE_TIMEOUT, provider.fetch_model_info()).await {
        // A response carrying usable data (a window and/or effort levels) is cached long.
        Ok(Ok(Some(info))) if info.context_window.is_some() || info.effort_levels.is_some() => {
            save(token_store, profile, model, &info, CacheSource::Api).await;
            Some(info)
        }
        // No usable metadata (an all-`None` response, or no row at all): short negative marker so a
        // recovered API / table fix supersedes it soon rather than re-probing every launch.
        Ok(Ok(Some(_) | None)) => {
            save_negative(token_store, profile, model).await;
            None
        }
        Ok(Err(error)) => {
            tracing::warn!("model metadata probe failed, using fallback: {}", error);
            save_negative(token_store, profile, model).await;
            None
        }
        Err(_elapsed) => {
            tracing::warn!(
                "model metadata probe timed out after {:?}, using fallback",
                PROBE_TIMEOUT
            );
            save_negative(token_store, profile, model).await;
            None
        }
    }
}

/// Resolve the context window without a network probe: config override > built-in table > fresh
/// cache > floor. Used where no provider handle is available (the REPL prompt gauge). The agent's
/// own [`resolve_model_metadata`] runs alongside and populates the cache, so an unknown model's
/// gauge converges to the accurate value by the next launch (and matches immediately for recognized
/// models). Effort is not resolved here; the gauge doesn't need it.
pub async fn resolve_context_window_cached(
    config_window: Option<u64>,
    token_store: &TokenStore,
    profile: Option<&str>,
    model: Option<&str>,
) -> u64 {
    let Some(model) = model else {
        return config_window.unwrap_or(DEFAULT_CONTEXT_WINDOW);
    };
    if let Some(window) = config_window.or_else(|| context_window_for_model(model)) {
        return window;
    }
    load_cached(token_store, profile, model)
        .await
        .and_then(|info| info.context_window)
        .unwrap_or(DEFAULT_CONTEXT_WINDOW)
}

/// Load a *fresh* cached [`ModelInfo`] (respecting the per-source TTL). `None` when there is no
/// row, the row is stale, or the read failed. No-op without a profile to key on.
async fn load_cached(
    token_store: &TokenStore,
    profile: Option<&str>,
    model: &str,
) -> Option<ModelInfo> {
    let profile = profile?;
    match token_store.load_model_metadata(profile, model).await {
        Ok(Some((info, source, age))) => {
            if age < CacheSource::ttl_for(&source).as_secs() as i64 {
                Some(info)
            } else {
                None
            }
        }
        Ok(None) => None,
        Err(error) => {
            tracing::debug!("model-metadata cache read failed: {}", error);
            None
        }
    }
}

/// Persist a resolved [`ModelInfo`] to the cache, best-effort. No-op without a profile to key on.
async fn save(
    token_store: &TokenStore,
    profile: Option<&str>,
    model: &str,
    info: &ModelInfo,
    source: CacheSource,
) {
    let Some(profile) = profile else { return };
    // A negative marker must not downgrade a still-fresh positive `api` row that a concurrent
    // process may have just written; a positive write always wins (`None`).
    let preserve_fresh_api =
        matches!(source, CacheSource::Resolver).then(|| API_TTL.as_secs() as i64);
    if let Err(error) = token_store
        .save_model_metadata(profile, model, info, source.as_str(), preserve_fresh_api)
        .await
    {
        tracing::debug!("model-metadata cache write failed: {}", error);
    }
}

/// Write the short-lived negative marker (an all-`None` [`ModelInfo`]) so neither the window nor
/// the effort re-probes on every launch after an unresolved model.
async fn save_negative(token_store: &TokenStore, profile: Option<&str>, model: &str) {
    let marker = ModelInfo {
        context_window: None,
        effort_levels: None,
    };
    save(token_store, profile, model, &marker, CacheSource::Resolver).await;
}

#[cfg(test)]
mod tests {
    use std::path::Path;

    use super::*;
    use crate::{provider::mock::MockProvider, session::SessionManager};

    async fn token_store() -> TokenStore {
        SessionManager::open(Some(Path::new(":memory:")))
            .await
            .expect("in-memory db")
            .token_store()
    }

    fn provider_with(context_window: Option<u64>) -> Arc<dyn Provider> {
        Arc::new(MockProvider::with_model_info(ModelInfo {
            context_window,
            effort_levels: None,
        }))
    }

    #[tokio::test]
    async fn config_override_wins_unprobed_and_uncached() {
        let store = token_store().await;
        let provider = provider_with(Some(999)); // would answer 999 if reached
        let resolved = resolve_model_metadata(
            Some(500_000),
            &provider,
            &store,
            Some("p"),
            Some("unknown-x"),
        )
        .await;
        assert_eq!(resolved.context_window, 500_000);
        assert!(
            store
                .load_model_metadata("p", "unknown-x")
                .await
                .unwrap()
                .is_none()
        );
    }

    #[tokio::test]
    async fn recognized_model_uses_table_not_provider() {
        let store = token_store().await;
        let provider = provider_with(Some(42)); // bogus; the table must win for a known model
        let resolved =
            resolve_model_metadata(None, &provider, &store, Some("p"), Some("gpt-5.6-sol")).await;
        assert_eq!(resolved.context_window, 1_050_000);
        // A no-catalog provider with a table-known window is not probed, so nothing is cached.
        assert!(
            store
                .load_model_metadata("p", "gpt-5.6-sol")
                .await
                .unwrap()
                .is_none()
        );
    }

    #[tokio::test]
    async fn catalog_provider_probes_even_when_table_known() {
        let store = token_store().await;
        // Codex-like: reports an effort catalog, so a table-known model is still probed so effort
        // is catalog-accurate. The probe result is cached even though the window comes from
        // the table.
        let provider = Arc::new(MockProvider::with_effort_catalog(ModelInfo {
            context_window: Some(1_050_000),
            effort_levels: Some(vec![
                "low".to_string(),
                "high".to_string(),
                "xhigh".to_string(),
            ]),
        })) as Arc<dyn Provider>;
        let resolved =
            resolve_model_metadata(None, &provider, &store, Some("p"), Some("gpt-5.6-sol")).await;
        // Window still comes from the table (authoritative for recognized models).
        assert_eq!(resolved.context_window, 1_050_000);
        // ...but the catalog was probed and cached (with its effort levels) for the effort default.
        let (cached, source, _) = store
            .load_model_metadata("p", "gpt-5.6-sol")
            .await
            .unwrap()
            .expect("catalog probe cached");
        assert_eq!(source, "api");
        assert_eq!(
            cached.effort_levels.as_deref(),
            Some(["low", "high", "xhigh"].map(String::from).as_slice())
        );
    }

    #[tokio::test]
    async fn api_probe_resolves_then_cache_serves() {
        let store = token_store().await;
        let provider = provider_with(Some(321_000));
        let resolved =
            resolve_model_metadata(None, &provider, &store, Some("p"), Some("unknown-y")).await;
        assert_eq!(resolved.context_window, 321_000);
        let (cached, source, _) = store
            .load_model_metadata("p", "unknown-y")
            .await
            .unwrap()
            .expect("api result cached");
        assert_eq!(cached.context_window, Some(321_000));
        assert_eq!(source, "api");
        // A second resolve serves the cache even if the provider would now answer differently.
        let provider2 = provider_with(Some(1));
        let resolved2 =
            resolve_model_metadata(None, &provider2, &store, Some("p"), Some("unknown-y")).await;
        assert_eq!(resolved2.context_window, 321_000);
    }

    #[tokio::test]
    async fn unknown_without_api_floors_and_negative_caches() {
        let store = token_store().await;
        let provider = provider_with(None); // provider reports nothing
        let resolved =
            resolve_model_metadata(None, &provider, &store, Some("p"), Some("unknown-z")).await;
        assert_eq!(resolved.context_window, DEFAULT_CONTEXT_WINDOW);
        let (cached, source, _) = store
            .load_model_metadata("p", "unknown-z")
            .await
            .unwrap()
            .expect("negative marker cached");
        assert_eq!(cached.context_window, None);
        assert_eq!(cached.effort_levels, None);
        assert_eq!(source, "resolver");
    }

    #[tokio::test]
    async fn cached_window_only_gauge_reads_offline() {
        let store = token_store().await;
        // Table-known model resolves offline with no cache write.
        assert_eq!(
            resolve_context_window_cached(None, &store, Some("p"), Some("gpt-5.6-sol")).await,
            1_050_000
        );
        // Unknown model with no cache floors.
        assert_eq!(
            resolve_context_window_cached(None, &store, Some("p"), Some("unknown-q")).await,
            DEFAULT_CONTEXT_WINDOW
        );
        // A prior probe's cache is served offline.
        let info = ModelInfo {
            context_window: Some(321_000),
            effort_levels: None,
        };
        store
            .save_model_metadata("p", "unknown-q", &info, "api", None)
            .await
            .unwrap();
        assert_eq!(
            resolve_context_window_cached(None, &store, Some("p"), Some("unknown-q")).await,
            321_000
        );
    }
}
