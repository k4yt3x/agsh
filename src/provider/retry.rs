//! Backoff policy for [`crate::error::MekaError::RetryableProvider`]. Pure, provider-agnostic: no
//! knowledge of streaming vs. non-streaming, no I/O. Callers (`agent.rs`) own the retry loop and
//! the decision of *whether* a given failure is safe to retry (e.g. no user-visible content shown
//! yet); this module only answers *how long to wait* before the next attempt.

use std::time::Duration;

/// Maximum number of retries after the initial attempt (so `MAX_PROVIDER_RETRIES + 1` total
/// attempts). Hardcoded, not config-exposed — matches the project's convention for turn-level retry
/// knobs (see `MAX_OVERFLOW_RETRIES` in `agent.rs`).
pub(crate) const MAX_PROVIDER_RETRIES: u32 = 2;

/// How long a retry sequence may have been running before a further attempt is refused, measured
/// from the first attempt.
///
/// The attempt cap alone bounds the number of tries, not what they cost. An attempt that fails fast
/// is cheap, but one that fails by running out `read_timeout` costs 300 seconds
/// ([`crate::provider::STREAM_IDLE_TIMEOUT`]), and three of those is fifteen minutes of a user
/// waiting on a turn that will fail anyway -- plus, on a non-streaming call, up to three
/// completions the provider may have generated and charged for.
///
/// One window, so a call that hung for its whole `read_timeout` is *not* tried again: it spends the
/// budget exactly, and `elapsed >= RETRY_BUDGET` refuses the next attempt. That is the deliberate
/// shape. Retrying is for a call that failed without costing much -- a reset connection, a refused
/// port, a body that stopped short -- and a provider that accepted the request and then went silent
/// for five minutes has already taken more of the user's turn than a second silence is worth.
///
/// It bounds when the next attempt may *start*, not what the sequence totals, and the difference is
/// worth stating rather than rounding off. `should_retry_provider_error` is asked only once an
/// attempt has already failed, so the sequence still costs whatever the attempt running when the
/// budget ran out cost. A failure just short of the window is the expensive case: one at 299
/// seconds permits a second attempt that may itself run the full 300, for about 600 in total. A
/// total is not available to bound, because a streaming attempt has no length of its own --
/// `read_timeout` resets on every event -- so the only honest ceiling is on starting another one.
///
/// Tied to that timeout rather than picked, so the two cannot drift. It is deliberately generous
/// against the ordinary case, where a rate-limited provider answers in milliseconds and the whole
/// sequence costs the 3 seconds of backoff.
///
/// This is the layer that can bound the cost. The classifier cannot: it sees one failure with no
/// idea how long the sequence has run, and an earlier attempt to bound it there by refusing to
/// retry timeouts made an undelivered request terminal (see
/// `crate::error::provider_transport_error`).
pub(crate) const RETRY_BUDGET: Duration = crate::provider::STREAM_IDLE_TIMEOUT;

/// Delay cap for the computed exponential backoff (no `Retry-After` header present).
///
/// Unreachable at the current [`MAX_PROVIDER_RETRIES`], which only ever asks for attempts 1 and 2
/// and so only ever gets 1s and 2s. It is kept as the guard it is: raising the retry count without
/// it would take the fourth wait to 8s, the sixth to 32s, and the tenth past eight minutes, and
/// nothing about `1u64 << attempt` announces that on the way past.
const BACKOFF_CAP: Duration = Duration::from_secs(8);

/// Delay cap for a provider-supplied `Retry-After` value.
///
/// Sized to honour the hint rather than to override it. Rate-limit windows in the wild are seconds
/// to a minute, and a cap below them turns "the provider told us exactly when to come back" into
/// "we came back too early and were refused again" -- which spends a retry to learn nothing. That
/// matters more at [`MAX_PROVIDER_RETRIES`] = 2 than it did at 3: there are only two to spend.
///
/// A cap is still needed, because `parse_retry_after` relays whatever the header said and the sleep
/// happens before the next budget check, so a broken or hostile upstream saying a day would be
/// obeyed. A minute bounds any single wait, and two of them are well inside [`RETRY_BUDGET`].
/// Ctrl-C works throughout regardless: the caller sleeps via `tokio::select!` against the turn's
/// cancellation token.
const RETRY_AFTER_CAP: Duration = Duration::from_secs(60);

/// How long a turn waits, once its retry sequence is spent, before sending the *unmodified* request
/// one last time rather than degrading its own content.
///
/// This is not another retry. It is the one attempt that can tell the two readings of a spent
/// budget apart. A 5xx answering a completion means either "the provider cannot handle this body"
/// or "the provider is having a moment", and the agent loop cannot see which; degrading answers the
/// first, and answering the second that way destroys content on the strength of a coincidence,
/// because the degraded retry then succeeds simply because the outage ended and
/// `TurnRecovery::persist_vindicated_repair` writes the loss to the store as proven-good. The whole
/// retry sequence is only [`MAX_PROVIDER_RETRIES`] attempts across three seconds of backoff, which
/// a routine `529` burst outlasts easily, so without this the common case is the loss and the rare
/// case is the recovery.
///
/// Cheap in the case that matters and free in the case that does not: it is paid only by a turn
/// whose alternative was to start deleting things, and a payload that is genuinely broken costs one
/// wait before the degrade proceeds exactly as before.
///
/// [`BACKOFF_CAP`] rather than a number of its own, because that is already this module's answer to
/// "the longest a single wait is ever worth", and two constants would drift. It is a *floor*, not
/// the whole answer: see [`outage_reprieve`].
pub(crate) const OUTAGE_REPRIEVE: Duration = BACKOFF_CAP;

// `Duration::clamp` panics on an inverted range, and the floor and the ceiling in
// `outage_reprieve` are two independent constants that nothing else relates: `OUTAGE_REPRIEVE`
// tracks `BACKOFF_CAP`, tuned for interactive latency, while `RETRY_AFTER_CAP` bounds what a
// hostile upstream can ask for. Raising the first past the second is an ordinary-looking edit that
// would turn every `Retry-After` on a 5xx into an abort. A `const` assertion so it fails at compile
// time rather than on the one path a user reaches during an outage.
//
// A comment rather than a doc comment: this item is anonymous, so rustdoc renders nothing and a
// `///` here would silently swallow the next item's documentation instead. It did, for a while.
const _: () = assert!(OUTAGE_REPRIEVE.as_nanos() <= RETRY_AFTER_CAP.as_nanos());

/// How long the reprieve waits, given whatever the failing response asked for.
///
/// The provider's own `Retry-After` decides, up to [`RETRY_AFTER_CAP`], with [`OUTAGE_REPRIEVE`] as
/// the floor. Honouring it here rather than only in [`backoff_delay`] closes a gap that read badly
/// once stated: a `503` carrying `Retry-After: 60` had both retries wait the full minute on the
/// provider's instruction, and then the one wait that decides whether to *delete the user's
/// content* was eight seconds. The hint is the only evidence anyone has about how long the outage
/// lasts, and the wait it governs here is the most consequential of the three.
///
/// Still floored, because a provider that answers `Retry-After: 0` on a 5xx is not telling us the
/// outage is over; it is telling us nothing, and re-sending instantly would spend the reprieve
/// without giving the outage any time to end.
///
/// The `clamp` cannot panic: a `const` assertion above pins the floor at or below the ceiling.
pub(crate) fn outage_reprieve(retry_after: Option<Duration>) -> Duration {
    match retry_after {
        Some(hint) => hint.clamp(OUTAGE_REPRIEVE, RETRY_AFTER_CAP),
        None => OUTAGE_REPRIEVE,
    }
}

/// How long to wait before retry attempt number `attempt` (1-indexed: the first retry is
/// `attempt == 1`). Honors the provider's `Retry-After` hint when present (capped); otherwise
/// exponential backoff `1s, 2s, 4s, ...` capped at [`BACKOFF_CAP`], mirroring the shape of the
/// existing MCP reconnect backoff (`src/mcp.rs`) but tuned tighter for interactive turn latency.
pub(crate) fn backoff_delay(attempt: u32, retry_after: Option<Duration>) -> Duration {
    match retry_after {
        Some(delay) => delay.min(RETRY_AFTER_CAP),
        None => {
            let computed = Duration::from_secs(1u64 << attempt.saturating_sub(1));
            computed.min(BACKOFF_CAP)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The budget is the limit that binds, and it can only be that between two bounds.
    ///
    /// Below one attempt-window there is nothing left for a retry to spend, so no failure would
    /// ever be tried again and the whole mechanism would be dead. At `MAX_PROVIDER_RETRIES + 1`
    /// windows or more it could never fire before the attempt cap already had, so it would be
    /// decoration. [`RETRY_BUDGET`] currently sits at the low end of that range on purpose, which
    /// its own doc explains; what this pins is that a later edit to either constant keeps them in a
    /// relationship where the budget still does something.
    ///
    /// Worth asserting because the two constants live apart and neither reads the other's
    /// intent: `MAX_PROVIDER_RETRIES` could be raised to 8 and every behavioural test would still
    /// pass while the budget quietly became the only limit that ever fires.
    #[test]
    fn the_budget_can_still_be_the_limit_that_binds() {
        let window = crate::provider::STREAM_IDLE_TIMEOUT;
        let attempts = MAX_PROVIDER_RETRIES + 1;
        assert!(
            RETRY_BUDGET >= window,
            "a budget under one {window:?} window would refuse every retry"
        );
        assert!(
            RETRY_BUDGET < window * attempts,
            "a budget of {RETRY_BUDGET:?} can never fire before the {attempts}-attempt cap does"
        );
    }

    #[test]
    fn test_backoff_delay_exponential_without_retry_after() {
        assert_eq!(backoff_delay(1, None), Duration::from_secs(1));
        assert_eq!(backoff_delay(2, None), Duration::from_secs(2));
        assert_eq!(backoff_delay(3, None), Duration::from_secs(4));
        // Capped at BACKOFF_CAP (8s) even though 2^(4-1) = 8s exactly; confirm attempt 5 (16s
        // uncapped) is clamped too.
        assert_eq!(backoff_delay(4, None), Duration::from_secs(8));
        assert_eq!(backoff_delay(5, None), Duration::from_secs(8));
    }

    #[test]
    fn test_backoff_delay_zero_attempt_does_not_panic() {
        // `attempt.saturating_sub(1)` guards against underflow if ever called with 0.
        assert_eq!(backoff_delay(0, None), Duration::from_secs(1));
    }

    #[test]
    fn test_backoff_delay_uses_retry_after_when_present() {
        assert_eq!(
            backoff_delay(1, Some(Duration::from_secs(3))),
            Duration::from_secs(3)
        );
        // retry_after takes priority over the computed exponential value even at a later attempt.
        assert_eq!(
            backoff_delay(3, Some(Duration::from_secs(5))),
            Duration::from_secs(5)
        );
    }

    #[test]
    fn test_backoff_delay_caps_large_retry_after() {
        assert_eq!(
            backoff_delay(1, Some(Duration::from_secs(120))),
            RETRY_AFTER_CAP
        );
    }
}

#[cfg(test)]
mod reprieve_tests {
    use super::*;

    /// The reprieve waits as long as the provider asked, within the same bounds as a retry.
    ///
    /// It used to sleep the constant and drop `retry_after` on the floor, which read badly once
    /// stated: a `503` carrying `Retry-After: 60` had both retries wait the full minute on the
    /// provider's own instruction, and then the wait that decides whether to *delete the user's
    /// content* was eight seconds. All three bounds are asserted, because each exists for a
    /// different reason and a single-value test would pin none of them.
    #[test]
    fn the_reprieve_honours_the_hint_between_its_floor_and_the_shared_cap() {
        assert_eq!(
            outage_reprieve(None),
            OUTAGE_REPRIEVE,
            "no hint is the case the constant was chosen for"
        );
        assert_eq!(
            outage_reprieve(Some(Duration::from_secs(30))),
            Duration::from_secs(30),
            "a hint inside the bounds is the whole point"
        );
        assert_eq!(
            outage_reprieve(Some(Duration::from_secs(3600))),
            RETRY_AFTER_CAP,
            "and it is bounded by the same cap a retry is, not by the provider's patience"
        );
        assert_eq!(
            outage_reprieve(Some(Duration::ZERO)),
            OUTAGE_REPRIEVE,
            "`Retry-After: 0` on a 5xx says nothing; re-sending instantly would spend the \
             reprieve without giving the outage any time to end"
        );
    }
}
