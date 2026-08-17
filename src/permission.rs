//! Permission state machine governing what tools the agent may invoke. Levels: `none` (read-only,
//! no env info), `read` (filesystem reads), `ask` (writes prompt the user), `write` (writes go
//! through). The level is held in an [`AtomicU8`] so the REPL can mutate it concurrently with the
//! agent loop.

use std::{
    fmt,
    str::FromStr,
    sync::{
        Arc,
        atomic::{AtomicU8, Ordering},
    },
};

use crossterm::style::Color;

/// Permission levels in increasing order of autonomy (`None < Read < Ask < Write`). The derived
/// `Ord` follows the `#[repr(u8)]` discriminant and exists so a sub-agent's level can be clamped to
/// its parent's as a ceiling (`min`); it is deliberately distinct from [`Permission::allows`], a
/// capability predicate that treats `Ask` and `Write` as equal.
///
/// Serialises as the same lowercase word [`fmt::Display`] and [`FromStr`] use, so a persisted
/// permission reads the way it is written in `config.toml`.
#[derive(
    Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, serde::Serialize, serde::Deserialize,
)]
#[serde(rename_all = "lowercase")]
#[repr(u8)]
pub enum Permission {
    None = 0,
    Read = 1,
    Ask = 2,
    Write = 3,
}

impl Permission {
    pub fn cycle_next(self) -> Permission {
        match self {
            Permission::None => Permission::Read,
            Permission::Read => Permission::Ask,
            Permission::Ask => Permission::Write,
            Permission::Write => Permission::None,
        }
    }

    pub fn indicator(self) -> &'static str {
        match self {
            Permission::None => "n",
            Permission::Read => "r",
            Permission::Ask => "a",
            Permission::Write => "w",
        }
    }

    pub fn indicator_color(self) -> Color {
        match self {
            Permission::None => Color::Green,
            Permission::Read => Color::Yellow,
            Permission::Ask => Color::Magenta,
            Permission::Write => Color::Red,
        }
    }

    /// Returns true if this permission level allows using a tool that requires `required`.
    pub fn allows(self, required: Permission) -> bool {
        match self {
            Permission::None => required == Permission::None,
            Permission::Read => matches!(required, Permission::None | Permission::Read),
            Permission::Ask | Permission::Write => true,
        }
    }
}

impl fmt::Display for Permission {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Permission::None => write!(f, "none"),
            Permission::Read => write!(f, "read"),
            Permission::Ask => write!(f, "ask"),
            Permission::Write => write!(f, "write"),
        }
    }
}

impl FromStr for Permission {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "none" | "n" => Ok(Permission::None),
            "read" | "r" => Ok(Permission::Read),
            "ask" | "a" => Ok(Permission::Ask),
            "write" | "w" => Ok(Permission::Write),
            other => Err(format!(
                "invalid permission mode '{other}': expected 'none', 'read', 'ask', or 'write'"
            )),
        }
    }
}

/// Set of permission modes the user is allowed to switch into at runtime. Backed by a `u8` bitmask
/// indexed by [`Permission`]'s `repr(u8)` discriminant. Constructed via
/// [`EnabledPermissions::from_modes`] (or the constants); the constructor guarantees the set is
/// non-empty so [`Self::lowest`] is always well-defined.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EnabledPermissions {
    bits: u8,
}

impl EnabledPermissions {
    /// Every mode enabled. Used by test fixtures that don't care about the runtime gate; production
    /// code constructs the set from config.
    #[cfg(test)]
    pub const ALL: Self = Self { bits: 0b1111 };
    /// `none / read / write`. `ask` is opt-in.
    pub const DEFAULT: Self = Self {
        bits: (1 << Permission::None as u8)
            | (1 << Permission::Read as u8)
            | (1 << Permission::Write as u8),
    };

    /// Build an `EnabledPermissions` from any iterable of [`Permission`]s. Returns `None` if the
    /// iterator yields no items. An empty enabled set is meaningless (meka would have no level to
    /// start in), so the caller has to handle that case explicitly (typically by falling back to
    /// [`Self::DEFAULT`]).
    pub fn from_modes<I: IntoIterator<Item = Permission>>(iter: I) -> Option<Self> {
        let mut bits: u8 = 0;
        for mode in iter {
            bits |= 1 << (mode as u8);
        }
        if bits == 0 { None } else { Some(Self { bits }) }
    }

    pub fn is_enabled(self, mode: Permission) -> bool {
        self.bits & (1 << (mode as u8)) != 0
    }

    /// Iterate enabled modes in `none → read → ask → write` order.
    pub fn iter(self) -> impl Iterator<Item = Permission> {
        const ORDER: [Permission; 4] = [
            Permission::None,
            Permission::Read,
            Permission::Ask,
            Permission::Write,
        ];
        ORDER.into_iter().filter(move |&p| self.is_enabled(p))
    }

    /// Lowest-discriminant enabled mode. Every constructor ([`Self::from_modes`] returns `None`
    /// on empty input, [`Self::DEFAULT`] is non-empty by definition) refuses an empty set, so
    /// `iter().next()` always yields `Some`; the `expect` documents the invariant rather than
    /// checking it.
    #[allow(clippy::expect_used)]
    pub fn lowest(self) -> Permission {
        self.iter()
            .next()
            .expect("EnabledPermissions invariant: set is non-empty")
    }
}

/// The caller asked to switch to a mode that isn't in the enabled set.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct DisabledMode(pub Permission);

/// Lock-free shared handle to the current [`Permission`] level. Cloned freely across agent, REPL,
/// and tool-dispatch tasks. The REPL mutates this when the user cycles permission via `Shift+Tab`
/// or `/permission`; the dispatch loop reads it once at the enforcement site so mid-turn cycling
/// can't leave a tool acting on a stale snapshot.
#[derive(Clone)]
pub struct SharedPermission {
    inner: Arc<AtomicU8>,
    enabled: EnabledPermissions,
    /// A parent's live level, for a sub-agent's handle. [`Self::get`] returns the minimum of this
    /// and `inner`, so a parent downgrade takes effect on the worker's very next tool call.
    ///
    /// Without it the clamp happened only at spawn: `shared_permission` built a fresh `AtomicU8`
    /// from a snapshot of the parent's level, and nothing propagated afterwards. A user who
    /// pressed Shift+Tab to `none` to stop a runaway worker saw the prompt indicator change
    /// and the parent's next call denied, while the worker kept writing files and running
    /// unsandboxed commands to completion -- and `permissions.md` presents cycling the parent
    /// as the way to restrict sub-agents.
    ceiling: Option<Arc<AtomicU8>>,
}

impl SharedPermission {
    pub fn new(initial: Permission, enabled: EnabledPermissions) -> Self {
        Self {
            inner: Arc::new(AtomicU8::new(initial as u8)),
            enabled,
            ceiling: None,
        }
    }

    /// A handle bounded from above by `parent`'s live level, for a sub-agent.
    ///
    /// The child keeps its own level (it may sit below the parent, and `agent_spawn` clamps it at
    /// creation), but can never read higher than the parent does right now. A raise is inherited
    /// too, which is the same rule read in the other direction: the child's authority is always
    /// `min(its own grant, what the human currently permits)`.
    pub fn with_ceiling(
        initial: Permission,
        enabled: EnabledPermissions,
        parent: &SharedPermission,
    ) -> Self {
        Self {
            inner: Arc::new(AtomicU8::new(initial as u8)),
            enabled,
            // Share the parent's *own* cell rather than its effective value, and flatten a chain:
            // a grandchild whose parent is itself bounded takes the root's cell, and its own
            // spawn-time clamp already folded the intermediate level in. That keeps `get` a fixed
            // two loads however deep the tree goes.
            ceiling: Some(
                parent
                    .ceiling
                    .clone()
                    .unwrap_or_else(|| parent.inner.clone()),
            ),
        }
    }

    pub fn enabled(&self) -> EnabledPermissions {
        self.enabled
    }

    pub fn get(&self) -> Permission {
        let own = Self::decode(self.inner.load(Ordering::Relaxed));
        match &self.ceiling {
            Some(parent) => own.min(Self::decode(parent.load(Ordering::Relaxed))),
            None => own,
        }
    }

    fn decode(raw: u8) -> Permission {
        match raw {
            0 => Permission::None,
            1 => Permission::Read,
            2 => Permission::Ask,
            3 => Permission::Write,
            _ => Permission::None,
        }
    }

    /// Switch to `mode`. Returns `Err(DisabledMode(mode))` if the caller requested a mode that
    /// isn't in [`Self::enabled`]; the current level is left unchanged in that case.
    pub fn try_set(&self, mode: Permission) -> Result<(), DisabledMode> {
        if !self.enabled.is_enabled(mode) {
            return Err(DisabledMode(mode));
        }
        self.set_unchecked(mode);
        Ok(())
    }

    /// Low-level setter that bypasses the enabled-set check. Used by `try_set` / `cycle` and by
    /// tests that need to construct edge cases.
    pub(crate) fn set_unchecked(&self, mode: Permission) {
        self.inner.store(mode as u8, Ordering::Relaxed);
    }

    /// Advance to the next enabled mode in `none → read → ask → write → ...` order, skipping any
    /// disabled modes. If only one mode is enabled the cycle is a visual no-op (returns the current
    /// mode without changing it). Bounded to 4 iterations so it can never spin forever.
    pub fn cycle(&self) -> Permission {
        let mut next = self.get();
        for _ in 0..4 {
            next = next.cycle_next();
            if self.enabled.is_enabled(next) {
                self.set_unchecked(next);
                return next;
            }
        }
        // Unreachable when the constructor invariant holds (set non-empty), because the loop walks
        // through all four variants. Return current for safety instead of panicking.
        self.get()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_permission_allows() {
        assert!(Permission::Write.allows(Permission::None));
        assert!(Permission::Write.allows(Permission::Read));
        assert!(Permission::Write.allows(Permission::Ask));
        assert!(Permission::Write.allows(Permission::Write));

        assert!(Permission::Ask.allows(Permission::None));
        assert!(Permission::Ask.allows(Permission::Read));
        assert!(Permission::Ask.allows(Permission::Ask));
        assert!(Permission::Ask.allows(Permission::Write));

        assert!(Permission::Read.allows(Permission::None));
        assert!(Permission::Read.allows(Permission::Read));
        assert!(!Permission::Read.allows(Permission::Ask));
        assert!(!Permission::Read.allows(Permission::Write));

        assert!(Permission::None.allows(Permission::None));
        assert!(!Permission::None.allows(Permission::Read));
        assert!(!Permission::None.allows(Permission::Ask));
        assert!(!Permission::None.allows(Permission::Write));
    }

    #[test]
    fn test_permission_ordering() {
        assert!(Permission::None < Permission::Read);
        assert!(Permission::Read < Permission::Ask);
        assert!(Permission::Ask < Permission::Write);
        // `min` is the clamp used to bound a sub-agent's level to its parent's ceiling.
        assert_eq!(Permission::Read.min(Permission::Write), Permission::Read);
        assert_eq!(Permission::Write.min(Permission::Read), Permission::Read);
        assert_eq!(Permission::Ask.min(Permission::Write), Permission::Ask);
        assert_eq!(Permission::Write.min(Permission::Write), Permission::Write);
    }

    #[test]
    fn test_permission_cycle() {
        assert_eq!(Permission::None.cycle_next(), Permission::Read);
        assert_eq!(Permission::Read.cycle_next(), Permission::Ask);
        assert_eq!(Permission::Ask.cycle_next(), Permission::Write);
        assert_eq!(Permission::Write.cycle_next(), Permission::None);
    }

    #[test]
    fn test_permission_from_str() {
        assert_eq!(Permission::from_str("none"), Ok(Permission::None));
        assert_eq!(Permission::from_str("read"), Ok(Permission::Read));
        assert_eq!(Permission::from_str("ask"), Ok(Permission::Ask));
        assert_eq!(Permission::from_str("write"), Ok(Permission::Write));
        assert_eq!(Permission::from_str("n"), Ok(Permission::None));
        assert_eq!(Permission::from_str("r"), Ok(Permission::Read));
        assert_eq!(Permission::from_str("a"), Ok(Permission::Ask));
        assert_eq!(Permission::from_str("w"), Ok(Permission::Write));
        assert!(Permission::from_str("invalid").is_err());
    }

    #[test]
    fn test_permission_display() {
        assert_eq!(Permission::None.to_string(), "none");
        assert_eq!(Permission::Read.to_string(), "read");
        assert_eq!(Permission::Ask.to_string(), "ask");
        assert_eq!(Permission::Write.to_string(), "write");
    }

    #[test]
    fn test_enabled_permissions_default() {
        let default = EnabledPermissions::DEFAULT;
        assert!(default.is_enabled(Permission::None));
        assert!(default.is_enabled(Permission::Read));
        assert!(!default.is_enabled(Permission::Ask));
        assert!(default.is_enabled(Permission::Write));
        assert_eq!(default.iter().count(), 3);
    }

    #[test]
    fn test_enabled_permissions_all() {
        let all = EnabledPermissions::ALL;
        assert!(all.is_enabled(Permission::None));
        assert!(all.is_enabled(Permission::Read));
        assert!(all.is_enabled(Permission::Ask));
        assert!(all.is_enabled(Permission::Write));
        assert_eq!(all.iter().count(), 4);
    }

    #[test]
    fn test_enabled_permissions_from_modes() {
        let single = EnabledPermissions::from_modes([Permission::Read]).unwrap();
        assert!(single.is_enabled(Permission::Read));
        assert_eq!(single.iter().count(), 1);

        let dups =
            EnabledPermissions::from_modes([Permission::Read, Permission::Read, Permission::Write])
                .unwrap();
        assert_eq!(dups.iter().count(), 2);
        assert!(dups.is_enabled(Permission::Read));
        assert!(dups.is_enabled(Permission::Write));

        assert_eq!(EnabledPermissions::from_modes(std::iter::empty()), None);
    }

    #[test]
    fn test_enabled_permissions_iter_order() {
        let all = EnabledPermissions::ALL;
        let order: Vec<Permission> = all.iter().collect();
        assert_eq!(order, vec![
            Permission::None,
            Permission::Read,
            Permission::Ask,
            Permission::Write,
        ]);
    }

    #[test]
    fn test_enabled_permissions_lowest() {
        assert_eq!(EnabledPermissions::ALL.lowest(), Permission::None);
        assert_eq!(EnabledPermissions::DEFAULT.lowest(), Permission::None);
        assert_eq!(
            EnabledPermissions::from_modes([Permission::Ask, Permission::Write])
                .unwrap()
                .lowest(),
            Permission::Ask
        );
        assert_eq!(
            EnabledPermissions::from_modes([Permission::Write])
                .unwrap()
                .lowest(),
            Permission::Write
        );
    }

    #[test]
    fn test_shared_permission_basic() {
        let shared = SharedPermission::new(Permission::Read, EnabledPermissions::ALL);
        assert_eq!(shared.get(), Permission::Read);

        shared.try_set(Permission::Write).unwrap();
        assert_eq!(shared.get(), Permission::Write);
    }

    #[test]
    fn test_shared_permission_clone() {
        let shared = SharedPermission::new(Permission::Read, EnabledPermissions::ALL);
        let cloned = shared.clone();

        shared.try_set(Permission::Write).unwrap();
        assert_eq!(cloned.get(), Permission::Write);
    }

    #[test]
    fn test_shared_permission_try_set_disabled() {
        let shared = SharedPermission::new(Permission::Read, EnabledPermissions::DEFAULT);
        let err = shared.try_set(Permission::Ask).unwrap_err();
        assert_eq!(err.0, Permission::Ask);
        // Current mode unchanged.
        assert_eq!(shared.get(), Permission::Read);
    }

    #[test]
    fn test_shared_permission_cycle_skips_disabled() {
        let shared = SharedPermission::new(Permission::Read, EnabledPermissions::DEFAULT);
        // DEFAULT is none/read/write, so Read → Write (skips Ask).
        assert_eq!(shared.cycle(), Permission::Write);
        assert_eq!(shared.get(), Permission::Write);
        // Write → None.
        assert_eq!(shared.cycle(), Permission::None);
        // None → Read.
        assert_eq!(shared.cycle(), Permission::Read);
    }

    #[test]
    fn test_shared_permission_cycle_all_enabled() {
        let shared = SharedPermission::new(Permission::None, EnabledPermissions::ALL);
        assert_eq!(shared.cycle(), Permission::Read);
        assert_eq!(shared.cycle(), Permission::Ask);
        assert_eq!(shared.cycle(), Permission::Write);
        assert_eq!(shared.cycle(), Permission::None);
    }

    #[test]
    fn test_shared_permission_cycle_single_mode() {
        let only_read = EnabledPermissions::from_modes([Permission::Read]).unwrap();
        let shared = SharedPermission::new(Permission::Read, only_read);
        // Cycle returns the same mode and doesn't loop forever.
        assert_eq!(shared.cycle(), Permission::Read);
        assert_eq!(shared.get(), Permission::Read);
    }

    #[test]
    fn test_shared_permission_set_unchecked_bypasses_enabled() {
        // Used by tests that need to construct edge cases regardless of the configured enabled set
        // (e.g. prompt-cache invariance tests).
        let shared = SharedPermission::new(Permission::Read, EnabledPermissions::DEFAULT);
        shared.set_unchecked(Permission::Ask);
        assert_eq!(shared.get(), Permission::Ask);
    }
}
