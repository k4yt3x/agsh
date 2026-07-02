//! Per-session counters surfaced by `/status`. Shared across the agent (which records tokens and
//! turn count) and the Claude providers (which record image-redaction events).
//!
//! All fields are lock-free atomics so any task can update without contention; readers take a
//! [`SessionStatsSnapshot`] for display.

use std::sync::atomic::{AtomicU64, Ordering::Relaxed};

use crate::provider::TokenUsage;

#[derive(Debug, Default)]
pub struct SessionStats {
    turns: AtomicU64,
    input_tokens: AtomicU64,
    output_tokens: AtomicU64,
    cache_creation_input_tokens: AtomicU64,
    cache_read_input_tokens: AtomicU64,
    redactions: AtomicU64,
    redacted_images: AtomicU64,
    redacted_bytes: AtomicU64,
}

impl SessionStats {
    /// Rebuild the counters from a persisted snapshot, so a resumed session continues its lifetime
    /// totals instead of restarting at zero.
    pub fn from_snapshot(snapshot: &SessionStatsSnapshot) -> Self {
        Self {
            turns: AtomicU64::new(snapshot.turns),
            input_tokens: AtomicU64::new(snapshot.input_tokens),
            output_tokens: AtomicU64::new(snapshot.output_tokens),
            cache_creation_input_tokens: AtomicU64::new(snapshot.cache_creation_input_tokens),
            cache_read_input_tokens: AtomicU64::new(snapshot.cache_read_input_tokens),
            redactions: AtomicU64::new(snapshot.redactions),
            redacted_images: AtomicU64::new(snapshot.redacted_images),
            redacted_bytes: AtomicU64::new(snapshot.redacted_bytes),
        }
    }

    /// Roll a successful turn's usage into the running totals.
    pub fn record_turn(&self, usage: &TokenUsage) {
        self.turns.fetch_add(1, Relaxed);
        self.input_tokens.fetch_add(usage.input_tokens, Relaxed);
        self.output_tokens.fetch_add(usage.output_tokens, Relaxed);
        self.cache_creation_input_tokens
            .fetch_add(usage.cache_creation_input_tokens, Relaxed);
        self.cache_read_input_tokens
            .fetch_add(usage.cache_read_input_tokens, Relaxed);
    }

    /// Record a single body-redaction event from one of the Claude providers. Called when
    /// image-block redaction fires on an oversized request body.
    pub fn record_redaction(&self, images: u64, bytes: u64) {
        self.redactions.fetch_add(1, Relaxed);
        self.redacted_images.fetch_add(images, Relaxed);
        self.redacted_bytes.fetch_add(bytes, Relaxed);
    }

    pub fn snapshot(&self) -> SessionStatsSnapshot {
        SessionStatsSnapshot {
            turns: self.turns.load(Relaxed),
            input_tokens: self.input_tokens.load(Relaxed),
            output_tokens: self.output_tokens.load(Relaxed),
            cache_creation_input_tokens: self.cache_creation_input_tokens.load(Relaxed),
            cache_read_input_tokens: self.cache_read_input_tokens.load(Relaxed),
            redactions: self.redactions.load(Relaxed),
            redacted_images: self.redacted_images.load(Relaxed),
            redacted_bytes: self.redacted_bytes.load(Relaxed),
        }
    }
}

#[derive(Debug, Clone, Default, serde::Serialize, serde::Deserialize)]
pub struct SessionStatsSnapshot {
    pub turns: u64,
    pub input_tokens: u64,
    pub output_tokens: u64,
    pub cache_creation_input_tokens: u64,
    pub cache_read_input_tokens: u64,
    pub redactions: u64,
    pub redacted_images: u64,
    pub redacted_bytes: u64,
}

impl SessionStatsSnapshot {
    /// Sum of all three input-token tiers (live, cache-write, cache-read). Matches what Anthropic
    /// bills against "input".
    pub fn total_input_tokens(&self) -> u64 {
        self.input_tokens
            .saturating_add(self.cache_creation_input_tokens)
            .saturating_add(self.cache_read_input_tokens)
    }

    /// Cache-hit ratio as an integer percent (0–100). Returns 0 when no input tokens have been
    /// recorded yet.
    pub fn cache_hit_pct(&self) -> u64 {
        let total = self.total_input_tokens();
        if total == 0 {
            0
        } else {
            ((self.cache_read_input_tokens as f64) / (total as f64) * 100.0).round() as u64
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_turn_accumulates() {
        let stats = SessionStats::default();
        stats.record_turn(&TokenUsage {
            input_tokens: 100,
            output_tokens: 50,
            cache_creation_input_tokens: 20,
            cache_read_input_tokens: 30,
        });
        stats.record_turn(&TokenUsage {
            input_tokens: 5,
            output_tokens: 7,
            cache_creation_input_tokens: 1,
            cache_read_input_tokens: 2,
        });
        let snap = stats.snapshot();
        assert_eq!(snap.turns, 2);
        assert_eq!(snap.input_tokens, 105);
        assert_eq!(snap.output_tokens, 57);
        assert_eq!(snap.cache_creation_input_tokens, 21);
        assert_eq!(snap.cache_read_input_tokens, 32);
        assert_eq!(snap.total_input_tokens(), 105 + 21 + 32);
    }

    #[test]
    fn record_redaction_accumulates() {
        let stats = SessionStats::default();
        stats.record_redaction(2, 4_000_000);
        stats.record_redaction(1, 2_000_000);
        let snap = stats.snapshot();
        assert_eq!(snap.redactions, 2);
        assert_eq!(snap.redacted_images, 3);
        assert_eq!(snap.redacted_bytes, 6_000_000);
    }

    #[test]
    fn from_snapshot_seeds_all_fields() {
        // Resume rebuilds the live counters from the persisted snapshot.
        let snapshot = SessionStatsSnapshot {
            turns: 3,
            input_tokens: 10,
            output_tokens: 20,
            cache_creation_input_tokens: 30,
            cache_read_input_tokens: 40,
            redactions: 1,
            redacted_images: 2,
            redacted_bytes: 1024,
        };
        let round = SessionStats::from_snapshot(&snapshot).snapshot();
        assert_eq!(round.turns, 3);
        assert_eq!(round.input_tokens, 10);
        assert_eq!(round.output_tokens, 20);
        assert_eq!(round.cache_creation_input_tokens, 30);
        assert_eq!(round.cache_read_input_tokens, 40);
        assert_eq!(round.redactions, 1);
        assert_eq!(round.redacted_images, 2);
        assert_eq!(round.redacted_bytes, 1024);
        // A turn recorded after seeding accumulates on top of the seeded totals.
        let seeded = SessionStats::from_snapshot(&snapshot);
        seeded.record_turn(&TokenUsage {
            input_tokens: 5,
            output_tokens: 5,
            cache_creation_input_tokens: 0,
            cache_read_input_tokens: 0,
        });
        assert_eq!(seeded.snapshot().turns, 4);
        assert_eq!(seeded.snapshot().input_tokens, 15);
    }

    #[test]
    fn cache_hit_pct_zero_when_no_input() {
        let snap = SessionStats::default().snapshot();
        assert_eq!(snap.cache_hit_pct(), 0);
    }

    #[test]
    fn cache_hit_pct_rounds() {
        let stats = SessionStats::default();
        stats.record_turn(&TokenUsage {
            input_tokens: 10,
            output_tokens: 0,
            cache_creation_input_tokens: 5,
            cache_read_input_tokens: 85,
        });
        // total = 100, hit = 85
        assert_eq!(stats.snapshot().cache_hit_pct(), 85);
    }
}
