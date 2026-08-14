//! The scope catalogue and the check every authenticated handler runs against it.
//!
//! Scopes gate verbs, not rows: there are no tenants, so every configured token shares one session
//! namespace and a scope only decides which endpoints a token may reach. See
//! [`crate::server::auth`] for how a token resolves to a [`Principal`].
//!
//! The catalogue lives here rather than next to the router because two very distant places need to
//! agree on it: the handlers, which demand a scope, and config resolution, which warns about a
//! configured scope no handler will ever ask for.

use axum::http::StatusCode;

use super::{
    auth::Principal,
    errors::{ErrorKind, ProblemDetail},
};

/// Every scope meka recognises.
///
/// Session-scoped operations (turns, compaction, rewind, export, background tasks) sit under
/// `sessions:*`, because the thing being read or changed is one conversation. The stores meka owns
/// process-wide get their own pairs, so an operator can hand a bridge the ability to run turns
/// without also handing it the ability to empty the memory directory.
///
/// Kept sorted, and kept in lockstep with the scope table in the HTTP API docs and the `bearerAuth`
/// description in [`crate::server::openapi`].
pub const KNOWN_SCOPES: &[&str] = &[
    "mcp:r",
    "mcp:w",
    "memory:r",
    "memory:w",
    "schedule:r",
    "schedule:w",
    "sessions:r",
    "sessions:w",
    "skills:r",
    "skills:w",
];

/// Scopes that admit the server-level discovery endpoints (`/v1/info`, `/v1/instructions`,
/// `/v1/providers`, the skill palette).
///
/// Any read scope is enough, deliberately: a token configured with `sessions:r` so a client can
/// list sessions should also be able to see which model it is talking to, without an operator
/// having to also grant `mcp:r` and `skills:r` for what is not sensitive information.
pub const ANY_READ_SCOPES: &[&str] = &["sessions:r", "mcp:r", "skills:r", "memory:r", "schedule:r"];

// `ProblemDetail` is ~128 bytes and only constructed on the rejection path. Same trade-off as
// `extract_bearer` in auth.rs; see the rationale there.
#[allow(clippy::result_large_err)]
/// Require one named scope. The rejection names the missing scope, so a client that gets a 403
/// learns what to ask its operator for rather than having to diff against the docs.
pub fn require(principal: &Principal, scope: &str) -> Result<(), ProblemDetail> {
    if principal.has_scope(scope) {
        return Ok(());
    }
    Err(ProblemDetail::new(
        ErrorKind::AuthScope,
        StatusCode::FORBIDDEN,
        format!("scope `{}` is required", scope),
    ))
}

#[allow(clippy::result_large_err)]
/// Require at least one of `scopes`. Used by the discovery endpoints; see [`ANY_READ_SCOPES`].
pub fn require_any(principal: &Principal, scopes: &[&str]) -> Result<(), ProblemDetail> {
    if scopes.iter().any(|scope| principal.has_scope(scope)) {
        return Ok(());
    }
    let names = scopes
        .iter()
        .map(|scope| format!("`{}`", scope))
        .collect::<Vec<_>>()
        .join(", ");
    Err(ProblemDetail::new(
        ErrorKind::AuthScope,
        StatusCode::FORBIDDEN,
        format!("one of {} is required", names),
    ))
}

/// Warn about a configured scope no handler will ever demand.
///
/// A warning rather than a hard error, in both directions: rejecting would mean a config written
/// for a newer meka cannot start an older binary, and staying silent would mean a plausible typo
/// like `sessions:write` grants nothing at all while looking like it grants everything. Called once
/// per token at config-resolve time.
pub fn warn_unknown(scopes: &[String], token_description: Option<&str>) {
    for scope in scopes {
        if KNOWN_SCOPES.contains(&scope.as_str()) {
            continue;
        }
        match token_description {
            Some(description) => tracing::warn!(
                "serve token '{}' declares unknown scope '{}'; it grants nothing. Known scopes: {}",
                description,
                scope,
                KNOWN_SCOPES.join(", ")
            ),
            None => tracing::warn!(
                "a serve token declares unknown scope '{}'; it grants nothing. Known scopes: {}",
                scope,
                KNOWN_SCOPES.join(", ")
            ),
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    fn principal(scopes: &[&str]) -> Principal {
        Principal {
            token_id: "test".to_string(),
            scopes: scopes
                .iter()
                .map(|s| (*s).to_string())
                .collect::<Vec<_>>()
                .into(),
        }
    }

    #[test]
    fn require_admits_the_exact_scope_and_nothing_else() {
        let holder = principal(&["memory:r"]);
        assert!(require(&holder, "memory:r").is_ok());
        assert!(require(&holder, "memory:w").is_err());
        // Read does not imply write and write does not imply read: the catalogue is flat.
        let writer = principal(&["memory:w"]);
        assert!(require(&writer, "memory:r").is_err());
    }

    /// The 403 body has to name the scope, or a client holding the wrong token cannot tell which
    /// of several scopes an endpoint wanted.
    #[test]
    fn rejection_names_the_missing_scope() {
        let problem = require(&principal(&[]), "schedule:w").expect_err("no scopes held");
        assert_eq!(problem.status, 403);
        let detail = problem.detail.expect("detail is always set");
        assert!(detail.contains("schedule:w"), "{detail}");
    }

    #[test]
    fn require_any_admits_a_single_match() {
        assert!(require_any(&principal(&["skills:r"]), ANY_READ_SCOPES).is_ok());
        assert!(require_any(&principal(&["sessions:w"]), ANY_READ_SCOPES).is_err());
    }

    #[test]
    fn require_any_rejection_lists_every_candidate() {
        let problem = require_any(&principal(&[]), &["mcp:r", "mcp:w"]).expect_err("no scopes");
        let detail = problem.detail.expect("detail is always set");
        assert!(detail.contains("`mcp:r`"), "{detail}");
        assert!(detail.contains("`mcp:w`"), "{detail}");
    }

    /// Every scope a handler can demand must be in the catalogue, or config resolution warns about
    /// a scope that actually works. `ANY_READ_SCOPES` is the easiest one to forget when a new
    /// subsystem lands.
    #[test]
    fn any_read_scopes_are_all_catalogued() {
        for scope in ANY_READ_SCOPES {
            assert!(
                KNOWN_SCOPES.contains(scope),
                "'{scope}' is demanded by a handler but missing from KNOWN_SCOPES"
            );
        }
    }

    #[test]
    fn known_scopes_are_sorted_and_unique() {
        let mut sorted = KNOWN_SCOPES.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.as_slice(), KNOWN_SCOPES);
    }

    /// `warn_unknown` has no return value; this pins the classification it warns on so a rename of
    /// a real scope cannot silently start warning about every token that uses it.
    #[test]
    fn typo_scopes_are_not_in_the_catalogue() {
        for typo in ["sessions:write", "session:r", "skills", "memory:rw", ""] {
            assert!(!KNOWN_SCOPES.contains(&typo), "'{typo}' must not be known");
        }
    }

    #[test]
    fn principal_scopes_arc_is_cheap_to_clone() {
        // Guards the `Arc<[String]>` representation the middleware depends on for per-request
        // cloning; a switch to `Vec<String>` would silently make every request allocate.
        let holder = principal(&["sessions:r"]);
        let cloned = holder.clone();
        assert!(Arc::ptr_eq(&holder.scopes, &cloned.scopes));
    }
}
