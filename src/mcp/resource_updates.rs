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

/// Record that a resource was updated. Stamp is unix seconds.
pub fn record(server_name: &str, uri: &str) {
    let stamp = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    if let Ok(mut state) = ledger().lock() {
        state.insert((server_name.to_string(), uri.to_string()), stamp);
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
}
