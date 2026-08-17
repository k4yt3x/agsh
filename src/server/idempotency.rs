//! Stripe-style `Idempotency-Key` cache for blocking `POST /turn` responses.
//!
//! Key scope: `(token_id, key)`. Tokens get independent dedup namespaces. Body mismatch on
//! replay returns 409. Entries expire after 24 h and are pruned in the background.
//!
//! Concurrency: a per-key slot state machine (`Pending` → `Completed`) ensures only one
//! request executes; concurrent same-keyed requests see 409. Drop without commit clears the
//! `Pending` marker so retries aren't blocked.

use std::{
    collections::HashMap,
    sync::Arc,
    time::{Duration, Instant},
};

use sha2::{Digest, Sha256};
use tokio::sync::RwLock;

/// Cached envelope for a single completed request. The body is stored as JSON bytes so it can
/// be re-served byte-identical regardless of which serde-derived shape produced it.
#[derive(Debug, Clone)]
pub struct CachedResponse {
    pub status: u16,
    pub body: Vec<u8>,
    /// Hash of the original request body, for body-mismatch detection on replays.
    pub body_hash: [u8; 32],
    pub stored_at: Instant,
}

impl CachedResponse {
    pub fn is_expired(&self, ttl: Duration) -> bool {
        self.stored_at.elapsed() > ttl
    }
}

/// Slot state for a single `(token_id, key)` entry. Distinguishes "first request is running"
/// from "first request finished" so concurrent same-key requests can be rejected cleanly.
#[derive(Debug, Clone)]
enum Slot {
    /// First request observed this key; the handler holds an [`IdempotencyTicket`] that will
    /// either commit (→ `Cached`) or be dropped (→ entry removed). Carries the request's body
    /// hash so a same-key-different-body racing request can be told `409 Conflict` before its
    /// turn would even start.
    Pending {
        body_hash: [u8; 32],
        stored_at: Instant,
    },
    /// First request committed its response. Subsequent same-key + same-body requests get
    /// `Hit`; same-key + different-body requests get `Conflict`.
    Cached(CachedResponse),
}

impl Slot {
    fn is_expired(&self, ttl: Duration, pending_ttl: Duration) -> bool {
        match self {
            Slot::Pending { stored_at, .. } => stored_at.elapsed() > pending_ttl,
            Slot::Cached(entry) => entry.is_expired(ttl),
        }
    }
}

/// What identifies one idempotent request: the token that sent it, what it acts on, and the
/// client's key. Named because it appears in five signatures and reads as three anonymous strings
/// otherwise.
type CacheKey = (String, String, String);

/// Process-wide idempotency cache. The key is `(token_id, scope, idempotency_key)`; the value is a
/// `Slot` describing whether the first request is pending or completed.
///
/// `scope` is what the request acts on, which for a turn is the session id (empty when the turn
/// creates one). Without it, one client reusing an `Idempotency-Key` across two sessions -- the
/// natural thing to do when the key identifies *the client's* unit of work -- got the first
/// session's transcript back for the second session, and the second turn never ran.
///
/// `RwLock` over `Mutex` so read-only diagnostic queries can take a shared lock without
/// blocking writers.
#[derive(Clone)]
pub struct IdempotencyCache {
    inner: Arc<RwLock<HashMap<CacheKey, Slot>>>,
    /// TTL for `Cached` entries, Stripe's documented 24h.
    ttl: Duration,
    /// TTL for `Pending` entries, much shorter. If a handler holds a ticket longer than this
    /// (typically because it crashed or was abort()ed in a way that bypassed Drop), the
    /// pruner sweeps the entry so a retry can proceed. Longer than the longest reasonable
    /// turn duration to avoid premature eviction on slow legitimate turns.
    pending_ttl: Duration,
    /// Soft cap on the number of cached entries per `token_id`. When a token's entry count
    /// reaches the cap, the oldest `Cached` entry is evicted to make room for the new
    /// `Pending`. `Pending` entries are never evicted mid-flight: they're a contract with
    /// the in-flight ticket holder.
    ///
    /// Bounds the DoS surface from a malicious client sending many unique keys: the worst
    /// case is `max_entries_per_token` × (token count) cached entries plus in-flight
    /// `Pending`s. The full cap is per-token rather than global so a single misbehaving
    /// token can't push other tokens' entries out of the cache.
    max_entries_per_token: usize,
    /// Ceiling on the bytes one token's cached envelopes may occupy. The entry count alone did not
    /// bound memory: an envelope is a whole turn response, so a thousand of them at a few
    /// megabytes each is gigabytes held for the 24-hour TTL. When a commit would take a token
    /// past this, its oldest cached envelopes are dropped until it fits; a single envelope
    /// larger than the budget is not cached at all, and its retry re-executes, which is the
    /// correct outcome for a response nobody can afford to keep.
    max_bytes_per_token: usize,
}

impl IdempotencyCache {
    pub fn new(ttl: Duration) -> Self {
        Self {
            inner: Arc::new(RwLock::new(HashMap::new())),
            ttl,
            // 10 minutes, longer than any provider's max turn time, short enough that an
            // orphaned `Pending` doesn't block retries for hours.
            pending_ttl: Duration::from_secs(10 * 60),
            // 1000 entries per token covers any realistic retry pattern (Stripe-style clients
            // re-use keys for hours; 1000 unique keys in 24h is already an aggressive cadence).
            max_entries_per_token: 1000,
            // 64 MiB per token: room for a normal retry window of full transcripts, far below what
            // a thousand large envelopes would have held.
            max_bytes_per_token: 64 * 1024 * 1024,
        }
    }

    /// Standard 24h TTL. Matches Stripe's documented guarantee for completed entries.
    pub fn standard() -> Self {
        Self::new(Duration::from_secs(24 * 60 * 60))
    }

    /// Atomically look up the slot for `(token_id, key)` and, if absent, reserve it by
    /// inserting a `Pending` marker. The returned `LookupOutcome` reflects what the caller
    /// should do:
    ///
    /// - `Hit(envelope)`: a previous request with the same key + same body completed; replay.
    /// - `Conflict`: a previous request with the same key but a different body completed (or is
    ///   pending). Return 409 `idempotency-conflict`.
    /// - `InFlight`: another request with the same key + same body is currently running. Return 409
    ///   `idempotency-conflict` (with a different `detail` message). Clients can retry after a
    ///   brief delay.
    /// - `Miss(ticket)`: no prior request; the caller now owns the slot via the ticket. Must call
    ///   `ticket.commit(...)` on completion to upgrade the slot to `Cached`; drop without commit
    ///   removes the `Pending` entry so retries aren't blocked forever.
    pub async fn lookup_and_mark(
        &self,
        token_id: &str,
        scope: &str,
        key: &str,
        body_hash: &[u8; 32],
    ) -> LookupOutcome {
        let composite_key = (token_id.to_string(), scope.to_string(), key.to_string());
        let mut inner = self.inner.write().await;
        // Take the slot (if any) for in-place inspection.
        match inner.get(&composite_key) {
            Some(slot) if slot.is_expired(self.ttl, self.pending_ttl) => {
                // Expired: fall through and treat as absent.
                inner.remove(&composite_key);
            }
            Some(Slot::Cached(entry)) => {
                if &entry.body_hash != body_hash {
                    return LookupOutcome::Conflict;
                }
                return LookupOutcome::Hit(entry.clone());
            }
            Some(Slot::Pending {
                body_hash: pending_hash,
                ..
            }) => {
                if pending_hash != body_hash {
                    return LookupOutcome::Conflict;
                }
                return LookupOutcome::InFlight;
            }
            None => {}
        }
        // Enforce per-token cap: evict the oldest `Cached` to make room. `Pending` entries are
        // never evicted: they're owed to the in-flight ticket holder. The common path (under
        // the cap) only counts; the scan-and-evict pass runs solely when the cap is hit.
        let token_count = inner.keys().filter(|(tid, ..)| tid == token_id).count();
        if token_count >= self.max_entries_per_token {
            let victim = inner
                .iter()
                .filter_map(|(key, slot)| match slot {
                    Slot::Cached(entry) if key.0 == token_id => Some((entry.stored_at, key)),
                    _ => None,
                })
                .min_by_key(|(stored_at, _)| *stored_at)
                .map(|(_, key)| key.clone());
            match victim {
                Some(victim) => {
                    inner.remove(&victim);
                }
                None => {
                    // All `max_entries_per_token` slots are `Pending`. Refuse the new key:
                    // evicting one would break a still-in-flight entry's contract. Surface as a
                    // 429 back-pressure signal so the client slows its unique-key cadence.
                    return LookupOutcome::CapExceeded;
                }
            }
        }
        // Reserve the slot and hand the caller a ticket.
        inner.insert(composite_key.clone(), Slot::Pending {
            body_hash: *body_hash,
            stored_at: Instant::now(),
        });
        drop(inner);
        LookupOutcome::Miss(IdempotencyTicket {
            cache: self.clone(),
            key: composite_key,
            body_hash: *body_hash,
            committed: false,
        })
    }

    /// Replace the `Pending` slot at `key` with a `Cached` envelope. Called by
    /// `IdempotencyTicket::commit`. Not public: callers acquire a ticket via `lookup_and_mark`.
    async fn commit(&self, key: CacheKey, body_hash: [u8; 32], status: u16, body: Vec<u8>) {
        let mut inner = self.inner.write().await;
        if body.len() > self.max_bytes_per_token {
            tracing::debug!(
                "idempotency envelope of {} bytes exceeds the {} byte per-token budget; not \
                 caching it, so a retry re-executes",
                body.len(),
                self.max_bytes_per_token
            );
            inner.remove(&key);
            return;
        }

        // Make room before inserting, oldest first. Only `Cached` entries are evictable: a
        // `Pending` is a contract with an in-flight ticket holder.
        let token_id = key.0.clone();
        let mut used: usize = inner
            .iter()
            .filter(|(other, _)| other.0 == token_id)
            .filter_map(|(_, slot)| match slot {
                Slot::Cached(entry) => Some(entry.body.len()),
                Slot::Pending { .. } => None,
            })
            .sum();
        while used + body.len() > self.max_bytes_per_token {
            let oldest = inner
                .iter()
                .filter(|(other, _)| other.0 == token_id && **other != key)
                .filter_map(|(other, slot)| match slot {
                    Slot::Cached(entry) => Some((entry.stored_at, other.clone(), entry.body.len())),
                    Slot::Pending { .. } => None,
                })
                .min_by_key(|(stored_at, ..)| *stored_at);
            match oldest {
                Some((_, victim, freed)) => {
                    inner.remove(&victim);
                    used = used.saturating_sub(freed);
                }
                // Nothing left to evict, which means this token holds only pending entries.
                None => break,
            }
        }

        inner.insert(
            key,
            Slot::Cached(CachedResponse {
                status,
                body,
                body_hash,
                stored_at: Instant::now(),
            }),
        );
    }

    /// Remove the `Pending` slot at `key` without recording a cached envelope. Called by
    /// `IdempotencyTicket::drop` when the handler returned without committing, typically
    /// because of a panic or an early abort. Removing the entry lets retries proceed.
    async fn rollback(&self, key: CacheKey) {
        let mut inner = self.inner.write().await;
        if let Some(Slot::Pending { .. }) = inner.get(&key) {
            inner.remove(&key);
        }
    }

    /// Background pruning task that removes expired entries periodically. Spawned at server
    /// startup; cancel by aborting the returned handle.
    pub fn spawn_pruner(&self) -> tokio::task::JoinHandle<()> {
        let inner = self.inner.clone();
        let ttl = self.ttl;
        let pending_ttl = self.pending_ttl;
        // Scan every 5 minutes: short enough that expired entries don't sit forever, long
        // enough that the lock contention is negligible.
        let scan = Duration::from_secs(5 * 60);
        tokio::spawn(async move {
            let mut ticker = tokio::time::interval(scan);
            ticker.tick().await;
            loop {
                ticker.tick().await;
                // Supervised like the other resident loops. A panic here is unlikely -- the body is
                // a `retain` over owned data -- but the consequence is the same shape as the
                // others: the task dies, nothing joins it, and the cache grows without bound for
                // the life of the process with no symptom until the memory shows.
                let prune = std::panic::AssertUnwindSafe(async {
                    let mut guard = inner.write().await;
                    let before = guard.len();
                    guard.retain(|_, slot| !slot.is_expired(ttl, pending_ttl));
                    before.saturating_sub(guard.len())
                });
                match futures::FutureExt::catch_unwind(prune).await {
                    Ok(pruned) if pruned > 0 => {
                        tracing::debug!("idempotency cache: pruned {} expired entries", pruned);
                    }
                    Ok(_) => {}
                    Err(panic) => tracing::error!(
                        "idempotency cache prune panicked ({}); continuing",
                        crate::error::panic_message(&*panic)
                    ),
                }
            }
        })
    }
}

/// RAII guard handed to the first caller that observes a miss for a given `(token_id, key)`.
/// The slot is held in `Pending` state for as long as the ticket is alive; calling
/// `commit(status, body)` upgrades to `Cached`. Dropping without committing removes the
/// `Pending` entry so retries are unblocked.
///
/// `commit` is async; `Drop` cannot await, so the rollback path uses `tokio::spawn` to enqueue
/// the cleanup. The cleanup task runs after the response has flushed and only touches the
/// cache map: no user-observable effect.
pub struct IdempotencyTicket {
    cache: IdempotencyCache,
    key: CacheKey,
    body_hash: [u8; 32],
    committed: bool,
}

impl std::fmt::Debug for IdempotencyTicket {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("IdempotencyTicket")
            .field("token", &self.key.0)
            .field("key", &self.key.1)
            .field("committed", &self.committed)
            .finish()
    }
}

impl IdempotencyTicket {
    /// Replace the reserved `Pending` slot with a `Cached` envelope. Marks the ticket
    /// committed so `Drop` skips the rollback.
    pub async fn commit(mut self, status: u16, body: Vec<u8>) {
        self.cache
            .commit(self.key.clone(), self.body_hash, status, body)
            .await;
        self.committed = true;
    }
}

impl Drop for IdempotencyTicket {
    fn drop(&mut self) {
        if self.committed {
            return;
        }
        // `tokio::spawn` panics if no runtime is bound (e.g. during shutdown). Guard with
        // `Handle::try_current` so a ticket dropped at exit-time doesn't crash the process.
        let Ok(handle) = tokio::runtime::Handle::try_current() else {
            tracing::debug!(
                "idempotency ticket dropped without a tokio runtime; skipping rollback (likely shutdown)"
            );
            return;
        };
        let cache = self.cache.clone();
        let key = self.key.clone();
        handle.spawn(async move {
            cache.rollback(key).await;
        });
    }
}

#[derive(Debug)]
pub enum LookupOutcome {
    Hit(CachedResponse),
    Conflict,
    InFlight,
    Miss(IdempotencyTicket),
    /// Per-token entry cap reached and only `Pending` slots are available to evict. Surface to
    /// the client as 429 `idempotency` (cache capacity) with a `Retry-After` hint; they should
    /// slow down the unique-key cadence.
    CapExceeded,
}

/// Compute the body hash used as the cache's tamper-detection signal. SHA-256 over the raw
/// request bytes: clients that re-send byte-identical bodies get a hit, anything else gets
/// a conflict.
pub fn hash_body(body: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(body);
    hasher.finalize().into()
}

#[cfg(test)]
mod tests {
    use super::*;

    /// An `Idempotency-Key` names the client's unit of work, so reusing one across two sessions is
    /// the natural thing to do. Without the session in the key, the second request replayed the
    /// first session's transcript and its turn never ran.
    #[tokio::test]
    async fn the_same_key_against_two_sessions_does_not_cross_answer() {
        let cache = IdempotencyCache::new(Duration::from_secs(60));
        let hash = hash_body(b"same body");

        let ticket = match cache
            .lookup_and_mark("token", "session-a", "k", &hash)
            .await
        {
            LookupOutcome::Miss(ticket) => ticket,
            other => panic!("expected a miss, got {:?}", other),
        };
        ticket.commit(200, b"session-a transcript".to_vec()).await;

        // The other session must run its own turn, not be handed session A's answer.
        match cache
            .lookup_and_mark("token", "session-b", "k", &hash)
            .await
        {
            LookupOutcome::Miss(_) => {}
            other => panic!("expected a miss for the second session, got {:?}", other),
        }

        // A genuine replay against the same session still hits.
        match cache
            .lookup_and_mark("token", "session-a", "k", &hash)
            .await
        {
            LookupOutcome::Hit(entry) => {
                assert_eq!(entry.body, b"session-a transcript");
            }
            other => panic!("expected a hit for the same session, got {:?}", other),
        }
    }

    /// An envelope is a whole turn response, so a thousand of them is memory the count cap alone
    /// never bounded.
    #[tokio::test]
    async fn a_token_cannot_hold_more_than_its_byte_budget() {
        let mut cache = IdempotencyCache::new(Duration::from_secs(60));
        cache.max_bytes_per_token = 1000;

        for index in 0..5 {
            let key = format!("k{index}");
            let hash = hash_body(key.as_bytes());
            let ticket = match cache.lookup_and_mark("token", "s", &key, &hash).await {
                LookupOutcome::Miss(ticket) => ticket,
                other => panic!("expected a miss, got {:?}", other),
            };
            ticket.commit(200, vec![b'x'; 400]).await;
        }

        let held: usize = cache
            .inner
            .read()
            .await
            .values()
            .filter_map(|slot| match slot {
                Slot::Cached(entry) => Some(entry.body.len()),
                Slot::Pending { .. } => None,
            })
            .sum();
        assert!(held <= 1000, "{held} bytes held, budget is 1000");
    }

    #[tokio::test]
    async fn hit_returns_cached_response_after_commit() {
        let cache = IdempotencyCache::standard();
        let body = b"{\"message\":\"hi\"}";
        let hash = hash_body(body);
        let ticket = match cache.lookup_and_mark("token1", "", "key-a", &hash).await {
            LookupOutcome::Miss(t) => t,
            other => panic!("expected Miss, got {:?}", other),
        };
        ticket.commit(200, b"response".to_vec()).await;
        match cache.lookup_and_mark("token1", "", "key-a", &hash).await {
            LookupOutcome::Hit(entry) => {
                assert_eq!(entry.status, 200);
                assert_eq!(entry.body, b"response");
            }
            other => panic!("expected Hit, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn body_mismatch_returns_conflict() {
        let cache = IdempotencyCache::standard();
        let original = hash_body(b"original");
        let different = hash_body(b"different");
        let ticket = match cache
            .lookup_and_mark("token1", "", "key-a", &original)
            .await
        {
            LookupOutcome::Miss(t) => t,
            _ => panic!("expected Miss"),
        };
        ticket.commit(200, b"response".to_vec()).await;
        assert!(matches!(
            cache
                .lookup_and_mark("token1", "", "key-a", &different)
                .await,
            LookupOutcome::Conflict
        ));
    }

    #[tokio::test]
    async fn concurrent_same_key_returns_in_flight() {
        let cache = IdempotencyCache::standard();
        let hash = hash_body(b"body");
        let _ticket = match cache.lookup_and_mark("token1", "", "key", &hash).await {
            LookupOutcome::Miss(t) => t,
            _ => panic!("expected Miss"),
        };
        // Second concurrent call with the same key + same body sees Pending, returns InFlight.
        assert!(matches!(
            cache.lookup_and_mark("token1", "", "key", &hash).await,
            LookupOutcome::InFlight
        ));
    }

    #[tokio::test]
    async fn concurrent_same_key_different_body_returns_conflict() {
        let cache = IdempotencyCache::standard();
        let _ticket = match cache
            .lookup_and_mark("token1", "", "key", &hash_body(b"a"))
            .await
        {
            LookupOutcome::Miss(t) => t,
            _ => panic!("expected Miss"),
        };
        assert!(matches!(
            cache
                .lookup_and_mark("token1", "", "key", &hash_body(b"b"))
                .await,
            LookupOutcome::Conflict
        ));
    }

    #[tokio::test]
    async fn ticket_drop_without_commit_unblocks_retries() {
        let cache = IdempotencyCache::standard();
        let hash = hash_body(b"body");
        let ticket = match cache.lookup_and_mark("token1", "", "key", &hash).await {
            LookupOutcome::Miss(t) => t,
            _ => panic!("expected Miss"),
        };
        // Simulate handler panic / abort: drop the ticket without commit.
        drop(ticket);
        // The drop spawns a cleanup task; yield to give it a chance to run.
        tokio::time::sleep(Duration::from_millis(20)).await;
        match cache.lookup_and_mark("token1", "", "key", &hash).await {
            LookupOutcome::Miss(t) => {
                // Retry succeeds: `Pending` was cleared by the drop.
                drop(t);
            }
            other => panic!("expected Miss after dropped ticket, got {:?}", other),
        }
    }

    #[tokio::test]
    async fn per_token_namespacing() {
        let cache = IdempotencyCache::standard();
        let hash = hash_body(b"body");
        let ticket = match cache.lookup_and_mark("token1", "", "key", &hash).await {
            LookupOutcome::Miss(t) => t,
            _ => panic!("expected Miss"),
        };
        ticket.commit(200, b"a".to_vec()).await;
        // token2 sees no entry: distinct namespaces.
        assert!(matches!(
            cache.lookup_and_mark("token2", "", "key", &hash).await,
            LookupOutcome::Miss(_),
        ));
    }

    #[tokio::test]
    async fn per_token_cap_evicts_oldest_cached_entry() {
        let mut cache = IdempotencyCache::standard();
        cache.max_entries_per_token = 3;

        // Fill the cap with three committed entries.
        for i in 0..3 {
            let key = format!("k{}", i);
            let hash = hash_body(format!("body-{}", i).as_bytes());
            let ticket = match cache.lookup_and_mark("token", "", &key, &hash).await {
                LookupOutcome::Miss(t) => t,
                _ => panic!("expected Miss for fresh key {}", i),
            };
            ticket
                .commit(200, format!("response-{}", i).into_bytes())
                .await;
            // Spread `stored_at` so LRU order is unambiguous.
            tokio::time::sleep(Duration::from_millis(5)).await;
        }

        // Fourth insert should succeed and evict `k0` (oldest).
        let hash = hash_body(b"body-3");
        let ticket = match cache.lookup_and_mark("token", "", "k3", &hash).await {
            LookupOutcome::Miss(t) => t,
            other => panic!("expected Miss after eviction, got {:?}", other),
        };
        ticket.commit(200, b"response-3".to_vec()).await;

        // `k0` is gone; `k1`, `k2`, `k3` remain.
        let h0 = hash_body(b"body-0");
        assert!(matches!(
            cache.lookup_and_mark("token", "", "k0", &h0).await,
            LookupOutcome::Miss(_),
        ));
    }

    #[tokio::test]
    async fn per_token_cap_refuses_when_all_pending() {
        let mut cache = IdempotencyCache::standard();
        cache.max_entries_per_token = 2;
        let _t1 = match cache
            .lookup_and_mark("token", "", "k1", &hash_body(b"a"))
            .await
        {
            LookupOutcome::Miss(t) => t,
            _ => panic!("k1 should miss"),
        };
        let _t2 = match cache
            .lookup_and_mark("token", "", "k2", &hash_body(b"b"))
            .await
        {
            LookupOutcome::Miss(t) => t,
            _ => panic!("k2 should miss"),
        };
        // Both tickets held → both slots Pending. Third request should be refused.
        assert!(matches!(
            cache
                .lookup_and_mark("token", "", "k3", &hash_body(b"c"))
                .await,
            LookupOutcome::CapExceeded,
        ));
    }

    #[tokio::test]
    async fn expired_cached_entry_is_a_miss() {
        let cache = IdempotencyCache::new(Duration::from_millis(1));
        let hash = hash_body(b"body");
        let ticket = match cache.lookup_and_mark("token1", "", "key", &hash).await {
            LookupOutcome::Miss(t) => t,
            _ => panic!("expected Miss"),
        };
        ticket.commit(200, b"a".to_vec()).await;
        tokio::time::sleep(Duration::from_millis(50)).await;
        assert!(matches!(
            cache.lookup_and_mark("token1", "", "key", &hash).await,
            LookupOutcome::Miss(_),
        ));
    }
}
