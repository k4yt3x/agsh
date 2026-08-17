//! In-memory ledger of resources that have been reported as changed via
//! `notifications/resources/updated`. The agent can query this via the `mcp_resource_updates_list`
//! builtin tool to see which resources need refreshing without subscribing again.

use std::{
    collections::HashMap,
    sync::{Mutex, OnceLock},
    time::{SystemTime, UNIX_EPOCH},
};

type Ledger = HashMap<(String /* server */, String /* uri */), u64>;

fn ledger() -> &'static Mutex<Ledger> {
    static STATE: OnceLock<Mutex<Ledger>> = OnceLock::new();
    STATE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// The most entries the ledger will hold. Keyed by `(server, uri)`, so a server that invents a
/// fresh URI per notification grows it without bound, and nothing ever removes an entry. The bound
/// is generous: a server with more than this many *distinct* resources changing in one process
/// lifetime is not one the agent can act on resource by resource anyway.
const MAX_LEDGER_ENTRIES: usize = 10_000;

/// Record that a resource was updated. Stamp is unix seconds.
pub fn record(server_name: &str, uri: &str) {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    if let Ok(mut state) = ledger().lock() {
        let key = (server_name.to_string(), uri.to_string());
        // Re-recording a URI already in the ledger is just a restamp and cannot grow it.
        if state.len() >= MAX_LEDGER_ENTRIES && !state.contains_key(&key) {
            // Drop the oldest rather than refusing the new one: the agent asks this list what to
            // re-read, and the freshest changes are the ones it has not seen.
            if let Some(oldest) = state
                .iter()
                .min_by_key(|(_, stamp)| **stamp)
                .map(|(key, _)| key.clone())
            {
                state.remove(&oldest);
                tracing::debug!(
                    "resource update ledger is full at {} entries; evicted {}:{}",
                    MAX_LEDGER_ENTRIES,
                    oldest.0,
                    oldest.1
                );
            }
        }
        state.insert(key, stamp);
    }
}

/// Snapshot every recorded update. Returned entries are sorted by server name then URI for stable
/// output.
pub fn snapshot() -> Vec<(String, String, u64)> {
    let Ok(state) = ledger().lock() else {
        return Vec::new();
    };
    let mut out: Vec<(String, String, u64)> = state
        .iter()
        .map(|((server, uri), stamp)| (server.clone(), uri.clone(), *stamp))
        .collect();
    out.sort_by(|a, b| (a.0.as_str(), a.1.as_str()).cmp(&(b.0.as_str(), b.1.as_str())));
    out
}

/// Drop every entry for a given server, used when the server is disconnected or removed via `meka
/// mcp remove`.
pub fn clear_for_server(server_name: &str) {
    if let Ok(mut state) = ledger().lock() {
        state.retain(|(name, _), _| name != server_name);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn record_and_snapshot() {
        record("srv", "file:///a");
        let snap = snapshot();
        assert!(snap.iter().any(|(s, u, _)| s == "srv" && u == "file:///a"));
    }

    #[test]
    fn clear_removes_matching() {
        record("srv-clear", "file:///b");
        clear_for_server("srv-clear");
        let snap = snapshot();
        assert!(!snap.iter().any(|(s, ..)| s == "srv-clear"));
    }

    /// The ledger is a process-lifetime map fed by a *server's* notifications, so an unbounded one
    /// is memory a remote peer decides the size of. Raising `MAX_LEDGER_ENTRIES` to `usize::MAX`
    /// left every suite green.
    ///
    /// Scoped to its own server name and cleaned up, because the ledger is process-global and the
    /// suite runs in parallel; the assertion is on the global total, which the cap governs.
    #[test]
    fn the_ledger_stops_growing_at_its_ceiling() {
        for index in 0..MAX_LEDGER_ENTRIES + 500 {
            record("srv-flood", &format!("file:///{}", index));
        }
        let total = snapshot().len();
        assert!(
            total <= MAX_LEDGER_ENTRIES,
            "the ledger grew past its ceiling: {total} > {MAX_LEDGER_ENTRIES}"
        );
        clear_for_server("srv-flood");
    }
}
