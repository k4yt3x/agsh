//! Bearer-token authentication for `meka serve`. Reads `Authorization: Bearer <token>` from each
//! incoming request, matches against the configured token list using constant-time comparison
//! (so a near-miss can't be distinguished from an early mismatch via timing), and attaches the
//! matched principal to the request extensions for downstream handlers to consult.
//!
//! No tenants: all configured tokens share the same session namespace. Scopes (`sessions:r`,
//! `sessions:w`, `skills:r`, `mcp:r`) gate which endpoints a token can hit. See the HTTP API docs
//! for the full scope catalogue.

use std::sync::Arc;

use axum::{
    extract::{Request, State},
    http::{HeaderMap, StatusCode, header},
    middleware::Next,
    response::Response,
};
use subtle::ConstantTimeEq;

use super::errors::{ErrorKind, ProblemDetail};
use crate::config::ResolvedServeToken;

/// Authenticated principal, attached as a request extension by [`bearer_auth`].
#[derive(Debug, Clone)]
pub struct Principal {
    /// Stable per-token identifier, the first 8 bytes of a SHA-256 of the token, hex-encoded.
    /// Used as the per-token key in the idempotency cache and structured logs (never the raw
    /// token, which we don't want appearing in observability output).
    pub token_id: String,
    pub scopes: Arc<[String]>,
}

impl Principal {
    /// Returns true iff this principal holds the named scope.
    pub fn has_scope(&self, required: &str) -> bool {
        self.scopes.iter().any(|s| s == required)
    }
}

/// One configured token paired with the `Principal` it resolves to. The fingerprint and scope
/// `Arc` are computed once at construction so per-request `lookup` only pays for the constant-time
/// compare plus a cheap `Principal` clone on match.
struct AuthEntry {
    token: String,
    principal: Principal,
}

/// Shared state the middleware reads on every request. Owned by the top-level `ServerState` and
/// cloned into the middleware via axum's `State` extractor.
#[derive(Clone)]
pub struct AuthRegistry {
    entries: Arc<[AuthEntry]>,
}

// Manual `Debug` so the raw token strings never reach logs: only the count and the non-secret
// fingerprints are printed, mirroring `ResolvedServeToken`'s redacting `Debug`.
impl std::fmt::Debug for AuthRegistry {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AuthRegistry")
            .field("token_count", &self.entries.len())
            .field(
                "token_ids",
                &self
                    .entries
                    .iter()
                    .map(|entry| entry.principal.token_id.as_str())
                    .collect::<Vec<_>>(),
            )
            .finish()
    }
}

impl AuthRegistry {
    pub fn new(tokens: Vec<ResolvedServeToken>) -> Self {
        let entries = tokens
            .into_iter()
            .map(|entry| AuthEntry {
                principal: Principal {
                    token_id: token_fingerprint(&entry.token),
                    scopes: entry.scopes.into(),
                },
                token: entry.token,
            })
            .collect();
        Self { entries }
    }

    /// Constant-time lookup. Iterates every configured token even on early match so a timing
    /// observer can't distinguish "matched index 0" from "matched index N" (or "no match"). The
    /// catch: with N tokens configured and constant-time compare, each request costs
    /// `O(N * token_length)` work. That's fine for realistic deployments (a handful of tokens);
    /// if you ever want hundreds, switch to a salted-hash lookup table.
    pub fn lookup(&self, presented: &str) -> Option<Principal> {
        let presented = presented.as_bytes();
        let mut matched: Option<&Principal> = None;
        for entry in self.entries.iter() {
            // ConstantTimeEq returns Choice::from(1u8) on match. Bytes-of-different-length never
            // match (no early return based on length difference).
            let candidate = entry.token.as_bytes();
            let is_match: bool = if candidate.len() == presented.len() {
                candidate.ct_eq(presented).into()
            } else {
                // Dummy ct_eq to equalise timing on length-mismatch.
                let _: bool = candidate.ct_eq(candidate).into();
                false
            };
            if is_match && matched.is_none() {
                matched = Some(&entry.principal);
            }
        }
        matched.cloned()
    }
}

/// Stable, non-reversible per-token identifier for logs and per-token cache keys. SHA-256
/// truncated to 8 bytes (16 hex chars), enough collision resistance for log correlation, never
/// long enough to reconstruct the token. `pub(crate)` so other modules (startup logging,
/// idempotency cache keying) share one source of truth for the fingerprint format.
pub(crate) fn token_fingerprint(token: &str) -> String {
    use sha2::{Digest, Sha256};
    let digest = Sha256::digest(token.as_bytes());
    let mut out = String::with_capacity(16);
    for byte in &digest[..8] {
        out.push_str(&format!("{:02x}", byte));
    }
    out
}

/// axum middleware: extracts the bearer token, looks it up in the registry, and attaches the
/// matched [`Principal`] to the request via `Extensions`. Routes that want to gate on a scope
/// extract the principal and call [`Principal::has_scope`] (or, preferred, use a small
/// extractor helper that does both in one step).
pub async fn bearer_auth(
    State(registry): State<AuthRegistry>,
    headers: HeaderMap,
    mut request: Request,
    next: Next,
) -> Response {
    let presented = match extract_bearer(&headers) {
        Ok(token) => token,
        Err(problem) => {
            tracing::debug!("auth rejected: {}", problem.title);
            return problem
                .instance(request.uri().path().to_string())
                .into_response();
        }
    };
    let Some(principal) = registry.lookup(&presented) else {
        let problem = ProblemDetail::new(
            ErrorKind::Auth,
            StatusCode::UNAUTHORIZED,
            "bearer token does not match any configured entry",
        )
        .instance(request.uri().path().to_string());
        return problem.into_response();
    };
    request.extensions_mut().insert(principal);
    next.run(request).await
}

// `ProblemDetail` is ~128 bytes (it carries several `String`s and a `BTreeMap`). The lint flags
// the `Err` variant as large; for an auth-failure path that fires at most once per request, the
// extra stack space is fine and boxing would just shuffle the same allocation under the heap.
fn extract_bearer(headers: &HeaderMap) -> Result<String, ProblemDetail> {
    let Some(value) = headers.get(header::AUTHORIZATION) else {
        return Err(ProblemDetail::new(
            ErrorKind::Auth,
            StatusCode::UNAUTHORIZED,
            "Authorization header is required",
        ));
    };
    let value = value.to_str().map_err(|_| {
        // Non-ASCII bytes → header is present but unparseable: "invalid", not "missing".
        ProblemDetail::new(
            ErrorKind::Auth,
            StatusCode::UNAUTHORIZED,
            "Authorization header is not a valid ASCII string",
        )
    })?;
    // RFC 7235 §2.1 says the auth-scheme is case-insensitive ("Bearer" / "bearer" / "BEARER" all
    // match). The scheme name must be followed by at least one whitespace character before the
    // token. We accept any leading whitespace before the scheme too; clients shouldn't send it,
    // but tolerating it is the kind thing.
    let trimmed = value.trim_start();
    let token = trimmed
        .get(..6)
        .filter(|prefix| prefix.eq_ignore_ascii_case("Bearer"))
        .and_then(|_| trimmed.get(6..))
        .and_then(|rest| {
            // The byte after "Bearer" must be ASCII whitespace per the grammar.
            rest.chars().next().filter(|c| c.is_ascii_whitespace())?;
            Some(rest.trim_start())
        });
    let Some(token) = token else {
        return Err(ProblemDetail::new(
            ErrorKind::Auth,
            StatusCode::UNAUTHORIZED,
            "Authorization header must use the `Bearer` scheme",
        ));
    };
    Ok(token.trim().to_string())
}

use axum::response::IntoResponse;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::TokenSource;

    fn entry(token: &str, scopes: &[&str]) -> ResolvedServeToken {
        ResolvedServeToken {
            token: token.to_string(),
            description: None,
            scopes: scopes.iter().map(|s| (*s).to_string()).collect(),
            source: TokenSource::Inline,
        }
    }

    #[test]
    fn lookup_matches_exact_token() {
        let registry = AuthRegistry::new(vec![entry("sk_test_a", &["sessions:r", "sessions:w"])]);
        let principal = registry.lookup("sk_test_a").expect("match");
        assert!(principal.has_scope("sessions:r"));
        assert!(principal.has_scope("sessions:w"));
        // Token_id is the fingerprint, not the raw token. Never log the raw value.
        assert_ne!(principal.token_id, "sk_test_a");
        assert_eq!(principal.token_id.len(), 16);
    }

    #[test]
    fn lookup_rejects_unknown_token() {
        let registry = AuthRegistry::new(vec![entry("sk_test_a", &["sessions:r"])]);
        assert!(registry.lookup("sk_test_b").is_none());
    }

    #[test]
    fn lookup_rejects_prefix_attack() {
        // Constant-time compare must reject a presented token that's a prefix of a configured
        // one (and vice versa); the length-mismatch branch in `lookup` handles this.
        let registry = AuthRegistry::new(vec![entry("sk_test_secret", &["sessions:r"])]);
        assert!(registry.lookup("sk_test_").is_none());
        assert!(registry.lookup("sk_test_secrets").is_none());
    }

    #[test]
    fn has_scope_exact_match() {
        let p = Principal {
            token_id: "abc".into(),
            scopes: vec!["sessions:r".into(), "mcp:r".into()].into(),
        };
        assert!(p.has_scope("sessions:r"));
        assert!(p.has_scope("mcp:r"));
        assert!(!p.has_scope("sessions:w"));
    }

    #[test]
    fn fingerprint_is_stable_and_truncated() {
        let a = token_fingerprint("sk_test");
        let b = token_fingerprint("sk_test");
        assert_eq!(a, b);
        assert_eq!(a.len(), 16);
        let c = token_fingerprint("sk_test_other");
        assert_ne!(a, c);
    }
}
