//! Permission state machine governing what tools the agent may invoke. Levels: `none` (read-only,
//! no env info), `read` (filesystem reads), `workspace` (writes confined to the workspace roots),
//! `ask` (writes reach anywhere but prompt the user), `unrestricted` (no boundary at all). The
//! level is held in an [`AtomicU8`] so the REPL can mutate it concurrently with the agent loop.
//!
//! `workspace` bounds *reach* while `ask` bounds *autonomy*, so the two are genuinely incomparable
//! and no total order over the five is honest. See [`Permission`] for how the three operations
//! defined here divide that up.

use std::{
    fmt,
    str::FromStr,
    sync::{
        Arc,
        atomic::{AtomicU8, Ordering},
    },
};

use crossterm::style::Color;

/// Permission levels ordered by *reach*: `None < Read < Workspace < Ask < Unrestricted`. `Ask`
/// outranks `Workspace` because an approved call at `ask` can write anywhere, while `workspace`
/// cannot leave its roots however many times it is invoked.
///
/// Three distinct operations are defined over these, and conflating any two of them is a bug:
///
/// - The derived `Ord` is the **display and cycle order** only. It is a total order over something
///   that is not totally ordered, so it must not be used to decide authority.
/// - [`Permission::allows`] is the **capability predicate**: may a tool requiring `required` be
///   dispatched at all. `Workspace`, `Ask` and `Unrestricted` are equal here, because scope is
///   enforced at the write door rather than by hiding tools. Keeping the tool set independent of
///   the level is what holds the API tools array byte-identical across mid-session toggles, which
///   the Claude prompt-cache prefix depends on.
/// - [`Permission::clamp_to`] is the **authority bound** for a sub-agent, and is the only one of
///   the three that models the partial order honestly.
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
    Workspace = 2,
    Ask = 3,
    Unrestricted = 4,
}

impl Permission {
    pub fn cycle_next(self) -> Permission {
        match self {
            Permission::None => Permission::Read,
            Permission::Read => Permission::Workspace,
            Permission::Workspace => Permission::Ask,
            Permission::Ask => Permission::Unrestricted,
            Permission::Unrestricted => Permission::None,
        }
    }

    /// Single-character prompt indicator. Every one is the mode name's first letter, which is what
    /// lets [`FromStr`] accept it as an alias without a second table to remember.
    pub fn indicator(self) -> &'static str {
        match self {
            Permission::None => "n",
            Permission::Read => "r",
            Permission::Workspace => "w",
            Permission::Ask => "a",
            Permission::Unrestricted => "u",
        }
    }

    /// Prompt-indicator colour. Green, yellow, orange and red form one temperature ramp over the
    /// three unattended modes plus `none`; magenta sits deliberately *off* that ramp because `ask`
    /// is off that axis, being the only mode with a human in the loop.
    ///
    /// Orange is spelled as truecolour rather than `DarkYellow`, which renders as olive or brown on
    /// most themes and would be confusable with the `Yellow` immediately below it in the cycle.
    /// `Color::Rgb` is already what the prompt line uses (`crate::config::default_input_style`), so
    /// this adds no assumption that was not being made one column to the left.
    pub fn indicator_color(self) -> Color {
        match self {
            Permission::None => Color::Green,
            Permission::Read => Color::Yellow,
            Permission::Workspace => Color::Rgb {
                r: 215,
                g: 135,
                b: 0,
            },
            Permission::Ask => Color::Magenta,
            Permission::Unrestricted => Color::Red,
        }
    }

    /// Returns true if this permission level allows using a tool that requires `required`.
    pub fn allows(self, required: Permission) -> bool {
        match self {
            Permission::None => required == Permission::None,
            Permission::Read => matches!(required, Permission::None | Permission::Read),
            Permission::Workspace | Permission::Ask | Permission::Unrestricted => true,
        }
    }

    /// Whether this level's authority is contained by `parent`'s, in *both* axes: how far writes
    /// reach, and whether a human sees them first.
    ///
    /// `Workspace` and `Ask` are mutually excluded here, and that is the whole point. `Ask` reaches
    /// further (an approved call writes anywhere) while `Workspace` is more autonomous (nothing is
    /// approved), so neither contains the other and any answer that says otherwise is granting
    /// something the parent does not hold.
    pub fn is_within(self, parent: Permission) -> bool {
        match self {
            Permission::None => true,
            Permission::Read => parent != Permission::None,
            Permission::Workspace => {
                matches!(parent, Permission::Workspace | Permission::Unrestricted)
            }
            Permission::Ask => matches!(parent, Permission::Ask | Permission::Unrestricted),
            Permission::Unrestricted => parent == Permission::Unrestricted,
        }
    }

    /// Whether this level may author a shell command that runs **unattended and unconfined**: a
    /// scheduled job's gate, which outlives the turn that created it and fires on a timer with
    /// nobody watching.
    ///
    /// `Unrestricted` alone. Two conditions have to hold at once and only the top rung satisfies
    /// both. Nobody is present at fire time, so `Ask` is out: its entire safety is a human
    /// answering a prompt. And a scheduled gate is spawned by `run_shell_probe` as a bare `sh -c`
    /// with no `Confinement`, no sandbox and meka's full environment, so the level that authorises
    /// it must be the one that promises no boundary.
    ///
    /// `Workspace` does not pass, tempting as it is on the reasoning that it is *safer* than the
    /// top rung. That is true of `execute_command`, which `workspace` confines, and false of a
    /// gate, which bypasses every backend. The result was a one-call escape: at `workspace`, a
    /// single `schedule_create` with a `gate` ran arbitrary commands outside the boundary within
    /// one poll interval, no race and no user interaction, while `execute_command` at the same
    /// level was confined and is refused outright when it cannot be. The interactive shell must not
    /// have a higher bar than the unattended one.
    ///
    /// A named predicate rather than a `matches!` repeated at each door, because the four sites had
    /// already drifted into phrasing the same rule three different ways.
    pub fn allows_unattended_shell(self) -> bool {
        matches!(self, Permission::Unrestricted)
    }

    /// Whether waking the agent unattended at this level could accomplish anything.
    ///
    /// Only `none` fails, and it fails completely: nothing is executable there, so a scheduled turn
    /// reads nothing, acts on nothing, and cannot even cancel the job that woke it. Registration is
    /// permission-independent, so the model does see the job in `[Scheduled]` and is offered
    /// `schedule_cancel`; the refusal happens at dispatch, which leaves it able to describe its
    /// predicament and unable to do anything about it. An `every = "5s"` job on such a session was
    /// a turn's worth of tokens every five seconds, forever, stoppable only by an operator.
    ///
    /// Distinct from [`Self::allows_unattended_shell`], which asks what a *gate* may run. This asks
    /// whether the job is worth running at all, so it applies to ungated jobs too.
    pub fn allows_unattended_work(self) -> bool {
        !matches!(self, Permission::None)
    }

    /// The highest level contained by **both** `self` and `other`.
    ///
    /// The meet, which `clamp_to` deliberately is not. `clamp_to` answers a *spawn* question ("the
    /// caller asked for this under that parent"), and its sideways move is argued there: a request
    /// that is not within the parent resolves to the parent's own level rather than collapsing to
    /// `read`.
    ///
    /// Replaying a **recorded** grant is a different question, and that argument does not carry.
    /// `agent_followup` re-clamps a stored `spec.permission` against the parent's *current* level,
    /// so a worker spawned at `workspace` under a parent since moved to `ask` came back at `ask` --
    /// whole-filesystem reach behind approvals, wider than the spawn call ever asked for. The spawn
    /// call is a statement about that worker, and a later parent change must not widen it.
    ///
    /// `workspace` and `ask` meet at `read`, which is the honest answer: neither contains the
    /// other, so the most either can safely share is what is under both.
    pub fn greatest_within_both(self, other: Permission) -> Permission {
        // Walked from the top so the *highest* satisfying level wins. Five rungs, so a scan is
        // clearer than an algebraic identity that a reader would have to re-derive.
        [
            Permission::Unrestricted,
            Permission::Ask,
            Permission::Workspace,
            Permission::Read,
            Permission::None,
        ]
        .into_iter()
        .find(|candidate| candidate.is_within(self) && candidate.is_within(other))
        .unwrap_or(Permission::None)
    }

    /// The authority a sub-agent actually gets when it asks for `self` under `parent`.
    ///
    /// Deliberately not `min`. The discriminants are a total order over a partially ordered set, so
    /// `min` leaks in whichever direction the ranking happens to fall: with `Workspace` below
    /// `Ask`, a parent at `ask` spawning a child at `workspace` would hand it *unattended* writes
    /// the parent itself does not have; ranked the other way, a parent at `workspace` spawning a
    /// child at `ask` would hand it the whole filesystem. Falling back to the parent's own level
    /// can do neither, and unlike a meet it never collapses an ordinary request down to `read`.
    ///
    /// One consequence is worth knowing because it looks like a bug and is not: this is **not
    /// monotone in `parent`**. Tightening a parent from `unrestricted` to `workspace` moves a child
    /// that asked for `ask` *sideways* to `workspace`, so a sub-agent that would stop and ask runs
    /// unattended instead. Nothing has escaped -- the child holds exactly the parent's authority,
    /// and its reach shrank from the whole filesystem to the workspace roots -- but the user
    /// tightened a setting and got fewer prompts, which is a real surprise. The alternative is the
    /// meet, `ask` under `workspace` giving `read`, which trades the surprise for a sub-agent that
    /// silently cannot write at all. Both are defensible; this one is chosen and pinned by
    /// `tightening_a_parent_can_move_an_ask_child_sideways_to_workspace`.
    pub fn clamp_to(self, parent: Permission) -> Permission {
        if self.is_within(parent) { self } else { parent }
    }
}

impl fmt::Display for Permission {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Permission::None => write!(f, "none"),
            Permission::Read => write!(f, "read"),
            Permission::Workspace => write!(f, "workspace"),
            Permission::Ask => write!(f, "ask"),
            Permission::Unrestricted => write!(f, "unrestricted"),
        }
    }
}

impl FromStr for Permission {
    type Err = String;

    fn from_str(s: &str) -> std::result::Result<Self, Self::Err> {
        match s.to_lowercase().as_str() {
            "none" | "n" => Ok(Permission::None),
            "read" | "r" => Ok(Permission::Read),
            "workspace" | "w" => Ok(Permission::Workspace),
            "ask" | "a" => Ok(Permission::Ask),
            "unrestricted" | "u" => Ok(Permission::Unrestricted),
            other => Err(format!(
                "invalid permission mode '{other}': expected 'none', 'read', 'workspace', 'ask', \
                 or 'unrestricted'"
            )),
        }
    }
}

/// Read a permission level back off a database row, given the subject to name if it cannot be read.
///
/// Returns `None` for both an absent column and an unreadable one, because every caller has the
/// same fallback for the two. What differs is that an unreadable value is *noticed*: spelling this
/// `.and_then(|value| value.parse().ok())` collapses "this session never recorded a level" into
/// "this session recorded a level meka cannot read" and resumes at the process default either way,
/// silently.
///
/// Repeats. The scheduler reads a job's session level every time that job comes due, so a row it
/// cannot read behind an `every = "1m"` watcher warns once a minute until the row is fixed. That is
/// deliberate rather than overlooked: the condition silently changes what a session runs at, and
/// the message names the session so it can be corrected.
pub fn parse_recorded_permission(
    recorded: Option<&str>,
    subject: &dyn fmt::Display,
) -> Option<Permission> {
    let raw = recorded?;
    match raw.parse() {
        Ok(permission) => Some(permission),
        Err(error) => {
            tracing::warn!(
                "{subject} records permission '{raw}', which is not one meka recognises ({error}); \
                 falling back to the configured level for this run"
            );
            None
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
    pub const ALL: Self = Self { bits: 0b11111 };
    /// `none / read / workspace / unrestricted`. `ask` is opt-in.
    ///
    /// `workspace` is in the default set because it is the rung a user reaching for "let the agent
    /// change things" should land on: Shift+Tab passes through it before `unrestricted`, so the
    /// confined mode is the one that costs fewer keystrokes.
    pub const DEFAULT: Self = Self {
        bits: (1 << Permission::None as u8)
            | (1 << Permission::Read as u8)
            | (1 << Permission::Workspace as u8)
            | (1 << Permission::Unrestricted as u8),
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

    /// Iterate enabled modes in `none → read → workspace → ask → unrestricted` order.
    pub fn iter(self) -> impl Iterator<Item = Permission> {
        const ORDER: [Permission; 5] = [
            Permission::None,
            Permission::Read,
            Permission::Workspace,
            Permission::Ask,
            Permission::Unrestricted,
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
    /// A parent's live level, for a sub-agent's handle. [`Self::get`] returns the greatest level
    /// within *both* this and `inner`, so a parent downgrade takes effect on the worker's very
    /// next tool call.
    ///
    /// Not the minimum, which is a different function on a partial order: `min` over the
    /// discriminants would hand a `workspace` child under an `ask` parent the `workspace` level,
    /// and the meet correctly gives `read`.
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
    /// creation), but can never reach further than the parent does right now. A raise is inherited
    /// too, which is the same rule read in the other direction: the child's authority is always
    /// [`Permission::greatest_within_both`] of its own grant and what the human currently permits.
    ///
    /// "Never further" is the honest phrasing; "never higher" was not, because the ladder is a
    /// partial order and the pair that matters is incomparable.
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
            //
            // That precondition is taken once, at spawn, and does not survive the root moving
            // afterwards. `a_grandchild_cannot_escape_an_intermediate_ask_parent` in
            // `crate::tools::subagent` states the residual window precisely under "What this does
            // not cover"; nothing exceeds the root, which is the human's own level, but the
            // direct-parent bound does not hold across a root that cycles between two spawns.
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
            // `greatest_within_both`, not `clamp_to` and certainly not `min`. The bound has to be
            // within the child's own grant *as well as* the parent's live level, because the two
            // are incomparable in one direction that matters.
            //
            // `clamp_to` was here, and it reproduced -- for a *running* worker -- exactly the
            // defect `SubagentSpec::effective_permission` was changed to stop. A worker spawned at
            // `workspace` keeps `Workspace` in its own cell; the user then moves the parent to
            // `ask`; `Workspace.is_within(Ask)` is false, so `clamp_to` returned the *parent's*
            // level and the worker became `Ask`. At `ask` `WriteScope::confined_to` yields `None`
            // and `Confinement::resolve` yields `Unconfined`, so the worker lost its write fence
            // and its shell sandbox at once -- strictly more reach than the `agent_spawn` call
            // asked for, granted by the user *tightening* their own level.
            //
            // The meet costs the already-accepted thing: a `workspace` worker under an `ask`
            // parent drops to `read`, because neither level contains the other and `read` is the
            // most either will vouch for.
            Some(parent) => own.greatest_within_both(Self::decode(parent.load(Ordering::Relaxed))),
            None => own,
        }
    }

    /// Decode a stored discriminant. An unrecognised byte falls to `None`, which is the safe
    /// direction: a corrupt or future value denies rather than grants.
    fn decode(raw: u8) -> Permission {
        match raw {
            0 => Permission::None,
            1 => Permission::Read,
            2 => Permission::Workspace,
            3 => Permission::Ask,
            4 => Permission::Unrestricted,
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

    /// Advance to the next enabled mode in `none → read → workspace → ask → unrestricted → ...`
    /// order, skipping any disabled modes. If only one mode is enabled the cycle is a visual no-op
    /// (returns the current mode without changing it). Bounded to 5 iterations so it can never spin
    /// forever.
    pub fn cycle(&self) -> Permission {
        let mut next = self.get();
        for _ in 0..5 {
            next = next.cycle_next();
            if self.enabled.is_enabled(next) {
                self.set_unchecked(next);
                return next;
            }
        }
        // Unreachable when the constructor invariant holds (set non-empty), because the loop walks
        // through all five variants. Return current for safety instead of panicking.
        self.get()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every level in the canonical order, for the exhaustive matrices below.
    const EVERY: [Permission; 5] = [
        Permission::None,
        Permission::Read,
        Permission::Workspace,
        Permission::Ask,
        Permission::Unrestricted,
    ];

    #[test]
    fn test_permission_allows() {
        assert!(Permission::Unrestricted.allows(Permission::None));
        assert!(Permission::Unrestricted.allows(Permission::Read));
        assert!(Permission::Unrestricted.allows(Permission::Ask));
        assert!(Permission::Unrestricted.allows(Permission::Unrestricted));

        assert!(Permission::Ask.allows(Permission::None));
        assert!(Permission::Ask.allows(Permission::Read));
        assert!(Permission::Ask.allows(Permission::Ask));
        assert!(Permission::Ask.allows(Permission::Unrestricted));

        assert!(Permission::Read.allows(Permission::None));
        assert!(Permission::Read.allows(Permission::Read));
        assert!(!Permission::Read.allows(Permission::Ask));
        assert!(!Permission::Read.allows(Permission::Unrestricted));

        assert!(Permission::None.allows(Permission::None));
        assert!(!Permission::None.allows(Permission::Read));
        assert!(!Permission::None.allows(Permission::Ask));
        assert!(!Permission::None.allows(Permission::Unrestricted));
    }

    /// `workspace` must dispatch every tool, including the ones that declare the rungs above it.
    ///
    /// Scope is enforced at the write door, not by withholding the tool, and this is also what
    /// keeps the API tools array byte-identical across mid-session toggles. A `workspace` that
    /// filtered the array would silently break the Claude prompt-cache prefix on every switch.
    #[test]
    fn workspace_dispatches_every_tool_and_read_still_does_not() {
        for required in EVERY {
            assert!(
                Permission::Workspace.allows(required),
                "workspace must dispatch a tool requiring {required}"
            );
        }
        assert!(Permission::Read.allows(Permission::Read));
        assert!(!Permission::Read.allows(Permission::Workspace));
    }

    #[test]
    fn test_permission_ordering() {
        assert!(Permission::None < Permission::Read);
        assert!(Permission::Read < Permission::Workspace);
        assert!(Permission::Workspace < Permission::Ask);
        assert!(Permission::Ask < Permission::Unrestricted);
    }

    /// The sub-agent clamp never returns authority the parent does not hold, for any of the 25
    /// pairs.
    ///
    /// The two that matter are the incomparable ones, and they are why this is not `min`. A parent
    /// at `ask` must not hand a child *unattended* workspace writes; a parent at `workspace` must
    /// not hand a child the whole filesystem. Whichever way a total order ranks the pair, `min`
    /// grants one of those.
    #[test]
    fn clamp_to_never_exceeds_the_parent_for_any_pair() {
        // The containment table, written out rather than derived.
        //
        // Asserting `requested.clamp_to(parent).is_within(parent)` is vacuous: it holds for *any*
        // reflexive `is_within`, because `clamp_to` returns either `self` (when it is already
        // within) or `parent` (which is within itself). Widening `Workspace.is_within(Ask)` to
        // `true` -- the dangerous direction, since it would hand a child unattended writes under a
        // parent whose safety is the prompt -- left that loop passing. Only the three hand-written
        // pairs below ever guarded anything.
        //
        // Rows are the child, columns the parent, in `EVERY` order: none, read, workspace, ask,
        // unrestricted.
        const CONTAINED: [[bool; 5]; 5] = [
            // none is within everything
            [true, true, true, true, true],
            // read is within anything except none
            [false, true, true, true, true],
            // workspace: itself and unrestricted. Not ask -- ask reaches further but is attended.
            [false, false, true, false, true],
            // ask: itself and unrestricted. Not workspace -- workspace is unattended.
            [false, false, false, true, true],
            // unrestricted is within itself alone
            [false, false, false, false, true],
        ];
        for (child_index, child) in EVERY.into_iter().enumerate() {
            for (parent_index, parent) in EVERY.into_iter().enumerate() {
                assert_eq!(
                    child.is_within(parent),
                    CONTAINED[child_index][parent_index],
                    "{child}.is_within({parent}) disagrees with the containment table"
                );
                // And the bound itself, which now rests on a table rather than on itself.
                let granted = child.clamp_to(parent);
                assert!(
                    CONTAINED[EVERY
                        .iter()
                        .position(|level| *level == granted)
                        .unwrap_or(0)][parent_index],
                    "{child} under {parent} yielded {granted}, which the table says is not within \
                     {parent}"
                );
            }
        }
        assert_eq!(
            Permission::Workspace.clamp_to(Permission::Ask),
            Permission::Ask,
            "a child asking for unattended workspace writes under an `ask` parent gets `ask`"
        );
        assert_eq!(
            Permission::Ask.clamp_to(Permission::Workspace),
            Permission::Workspace,
            "a child asking for whole-filesystem reach under a `workspace` parent gets `workspace`"
        );
        assert_eq!(
            Permission::Read.clamp_to(Permission::Unrestricted),
            Permission::Read,
            "an ordinary narrower request is granted as asked"
        );
    }

    /// A replayed grant is bounded by the spawn call as well as by the parent.
    ///
    /// The distinction this pins is the whole reason `greatest_within_both` exists alongside
    /// `clamp_to`. Spawn asks "what may this caller grant right now"; replay asks "what did the
    /// caller already grant, and does the parent still hold it". Using the spawn answer for replay
    /// let a worker recorded at `workspace` come back at `ask` when the parent moved -- reaching
    /// the whole filesystem, which the spawn call had explicitly declined.
    #[test]
    fn a_replayed_grant_is_never_wider_than_the_spawn_call_asked_for() {
        // The case that motivated it: neither contains the other, so they meet at `read`.
        assert_eq!(
            Permission::Workspace.greatest_within_both(Permission::Ask),
            Permission::Read
        );
        assert_eq!(
            Permission::Workspace.clamp_to(Permission::Ask),
            Permission::Ask,
            "spawn deliberately answers differently, and that is not a bug"
        );

        for recorded in EVERY {
            for parent in EVERY {
                let replayed = recorded.greatest_within_both(parent);
                assert!(
                    replayed.is_within(recorded),
                    "{recorded} replayed under {parent} gave {replayed}, wider than what was \
                     recorded"
                );
                assert!(
                    replayed.is_within(parent),
                    "{recorded} replayed under {parent} gave {replayed}, outside the parent"
                );
                // And it is the *greatest* such level, not merely a safe one: a worker must not be
                // crippled beyond what either bound requires.
                for candidate in EVERY {
                    if candidate.is_within(recorded) && candidate.is_within(parent) {
                        assert!(
                            candidate.is_within(replayed),
                            "{candidate} is within both {recorded} and {parent}, so {replayed} is \
                             not the greatest"
                        );
                    }
                }
            }
        }

        // Unchanged in the ordinary case: a parent that still holds the recorded level replays it
        // exactly.
        assert_eq!(
            Permission::Workspace.greatest_within_both(Permission::Unrestricted),
            Permission::Workspace
        );
        assert_eq!(
            Permission::Read.greatest_within_both(Permission::Read),
            Permission::Read
        );
    }

    /// Absent and unreadable are different answers, and the second one says so.
    ///
    /// Six call sites, no tests. The warn arm is reached by any row holding a value this build does
    /// not resolve: without the warning the level silently becomes the configured default and the
    /// only later clue is a message naming a level the row was never created at. Collapsing the two
    /// into a bare `None` is a one-character edit that nothing caught.
    #[test]
    fn an_unreadable_recorded_permission_warns_where_an_absent_one_is_silent() {
        #[derive(Clone)]
        struct Capture(std::sync::Arc<std::sync::Mutex<Vec<u8>>>);
        impl std::io::Write for Capture {
            fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
                self.0
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner())
                    .extend_from_slice(buf);
                Ok(buf.len())
            }

            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        impl<'a> tracing_subscriber::fmt::MakeWriter<'a> for Capture {
            type Writer = Self;

            fn make_writer(&'a self) -> Self::Writer {
                self.clone()
            }
        }

        let logged = |recorded: Option<&str>| -> (Option<Permission>, String) {
            let capture = Capture(std::sync::Arc::new(std::sync::Mutex::new(Vec::new())));
            let buffer = std::sync::Arc::clone(&capture.0);
            // Pinned to WARN because that is the default floor: a message emitted at `info` would
            // need `-v` to see, which is not a signal a user gets by default.
            let subscriber = tracing_subscriber::fmt()
                .with_writer(capture)
                .with_max_level(tracing::Level::WARN)
                .finish();
            let answer = tracing::subscriber::with_default(subscriber, || {
                parse_recorded_permission(recorded, &"job 7f3a1b2c")
            });
            let text = String::from_utf8_lossy(
                &buffer
                    .lock()
                    .unwrap_or_else(|poisoned| poisoned.into_inner()),
            )
            .into_owned();
            (answer, text)
        };

        // A column that was never written: not an error, and nothing to say about it.
        let (answer, text) = logged(None);
        assert_eq!(answer, None);
        assert!(text.is_empty(), "an absent level is not a warning: {text}");

        // A level meka still reads.
        let (answer, text) = logged(Some("workspace"));
        assert_eq!(answer, Some(Permission::Workspace));
        assert!(text.is_empty(), "a readable level is not a warning: {text}");

        // A value no build of meka resolves.
        let (answer, text) = logged(Some("elevated"));
        assert_eq!(answer, None, "an unreadable level must not resolve to one");
        assert!(
            text.contains("job 7f3a1b2c") && text.contains("elevated"),
            "the warning must name the subject and the unreadable value: {text}"
        );
    }

    /// A running worker keeps the fence its spawn call asked for when the parent *tightens* to a
    /// level that does not contain it.
    ///
    /// This is the live half of the rule `SubagentSpec::effective_permission` enforces at replay,
    /// and it was missing: `get` used `clamp_to`, which returns the *parent's* level for an
    /// incomparable pair. A worker spawned at `workspace` therefore became `ask` the moment the
    /// user moved the parent to `ask` -- and at `ask` there is no write fence and no shell sandbox
    /// at all, so tightening your own level handed the worker strictly more reach than the
    /// `agent_spawn({permission: "workspace"})` call had asked for.
    ///
    /// The whole suite passed with `clamp_to` here, which is why this asserts the worker's live
    /// level rather than the spec it was recorded with.
    #[test]
    fn tightening_a_parent_to_ask_does_not_unfence_a_workspace_worker() {
        let parent = SharedPermission::new(Permission::Unrestricted, EnabledPermissions::ALL);
        let worker =
            SharedPermission::with_ceiling(Permission::Workspace, EnabledPermissions::ALL, &parent);
        assert_eq!(worker.get(), Permission::Workspace, "the control");

        parent.set_unchecked(Permission::Ask);
        assert_eq!(
            worker.get(),
            Permission::Read,
            "a `workspace` worker under an `ask` parent must fall to the level both contain, not \
             rise to the parent's"
        );
        assert!(
            worker.get() != Permission::Ask,
            "`ask` has no write fence and no sandbox; reaching it is the escape this guards"
        );
    }

    /// Tightening the parent across the incomparable pair moves an `ask` child sideways to
    /// `workspace`, rather than down to the meet of the two, which would be `read`.
    ///
    /// The counterpart to `tightening_a_parent_to_ask_does_not_unfence_a_workspace_worker`: that
    /// one pins the direction a grant may not travel, this one pins the direction it may. The case
    /// for preferring this over the meet is argued on `clamp_to` itself.
    #[test]
    fn tightening_a_parent_can_move_an_ask_child_sideways_to_workspace() {
        assert_eq!(
            Permission::Ask.clamp_to(Permission::Unrestricted),
            Permission::Ask,
            "an `ask` child under an unrestricted parent keeps its prompts"
        );
        assert_eq!(
            Permission::Ask.clamp_to(Permission::Workspace),
            Permission::Workspace,
            "tightening that parent to `workspace` costs the child its prompts, and is still bound \
             by the parent"
        );
    }

    /// Only `unrestricted` may authorise a *shell* gate, and each exclusion is for its own reason.
    ///
    /// A gate whose probe is a tool call is authorised by
    /// [`crate::schedule::gate_probe_is_authorised`] instead, at the tool's own resolved level;
    /// this predicate answers only for the bare `sh -c` case, which is why it is the strict one.
    ///
    /// `ask` is out because a gate fires on a timer with nobody watching, so the prompt that is its
    /// entire safety will never be answered. `workspace` is out because a shell gate is spawned
    /// with no sandbox at all, so a level whose whole meaning is a write boundary cannot honestly
    /// authorise one. `workspace` must not pass here: it is a one-call escape, since
    /// `schedule_create` with a `gate` runs arbitrary unconfined commands from inside the confined
    /// mode.
    ///
    /// Spelled out per level rather than looped, because the two exclusions are different arguments
    /// and a future reader needs to see both before widening this.
    #[test]
    fn only_unrestricted_may_authorise_a_gate() {
        assert!(Permission::Unrestricted.allows_unattended_shell());
        assert!(
            !Permission::Workspace.allows_unattended_shell(),
            "a gate runs unsandboxed, so the confined level must not authorise one"
        );
        assert!(
            !Permission::Ask.allows_unattended_shell(),
            "nobody is present at fire time to answer the prompt `ask` relies on"
        );
        assert!(!Permission::Read.allows_unattended_shell());
        assert!(!Permission::None.allows_unattended_shell());
    }

    #[test]
    fn test_permission_cycle() {
        assert_eq!(Permission::None.cycle_next(), Permission::Read);
        assert_eq!(Permission::Read.cycle_next(), Permission::Workspace);
        assert_eq!(Permission::Workspace.cycle_next(), Permission::Ask);
        assert_eq!(Permission::Ask.cycle_next(), Permission::Unrestricted);
        assert_eq!(Permission::Unrestricted.cycle_next(), Permission::None);
    }

    #[test]
    fn test_permission_from_str() {
        assert_eq!(Permission::from_str("none"), Ok(Permission::None));
        assert_eq!(Permission::from_str("read"), Ok(Permission::Read));
        assert_eq!(Permission::from_str("workspace"), Ok(Permission::Workspace));
        assert_eq!(Permission::from_str("ask"), Ok(Permission::Ask));
        assert_eq!(
            Permission::from_str("unrestricted"),
            Ok(Permission::Unrestricted)
        );
        assert!(Permission::from_str("invalid").is_err());
    }

    /// Every indicator character round-trips through the parser, so what the prompt shows is always
    /// something `--permission`, `MEKA_PERMISSION` and `/permission` accept.
    #[test]
    fn every_indicator_character_parses_back_to_its_own_level() {
        for mode in EVERY {
            assert_eq!(
                Permission::from_str(mode.indicator()),
                Ok(mode),
                "indicator {:?} must parse back to {mode}",
                mode.indicator()
            );
        }
    }

    /// A mode `Permission` does not have is refused, and the refusal lists the ones it does.
    ///
    /// The list is the whole answer: meka names its five modes and does not try to guess which one
    /// an unrecognised string meant.
    #[test]
    fn an_unknown_mode_is_refused_and_the_refusal_names_the_five() {
        let error = Permission::from_str("write").expect_err("an unknown mode must not resolve");
        for mode in ["none", "read", "workspace", "ask", "unrestricted"] {
            assert!(
                error.contains(mode),
                "the refusal must name {mode}: {error}"
            );
        }
        assert_eq!(
            Permission::from_str("WRITE").expect_err("case-insensitively too"),
            error,
            "an unknown mode is matched after lowercasing, like every other alias"
        );
    }

    #[test]
    fn test_permission_display() {
        assert_eq!(Permission::None.to_string(), "none");
        assert_eq!(Permission::Read.to_string(), "read");
        assert_eq!(Permission::Workspace.to_string(), "workspace");
        assert_eq!(Permission::Ask.to_string(), "ask");
        assert_eq!(Permission::Unrestricted.to_string(), "unrestricted");
    }

    /// `workspace` ships enabled and `ask` does not.
    ///
    /// The pairing is the point: the confined rung is the one Shift+Tab reaches first, so a user
    /// who wants the agent to change things lands there rather than on the unbounded mode.
    #[test]
    fn test_enabled_permissions_default() {
        let default = EnabledPermissions::DEFAULT;
        assert!(default.is_enabled(Permission::None));
        assert!(default.is_enabled(Permission::Read));
        assert!(default.is_enabled(Permission::Workspace));
        assert!(!default.is_enabled(Permission::Ask));
        assert!(default.is_enabled(Permission::Unrestricted));
        assert_eq!(default.iter().count(), 4);
    }

    #[test]
    fn test_enabled_permissions_all() {
        let all = EnabledPermissions::ALL;
        for mode in EVERY {
            assert!(all.is_enabled(mode), "ALL must contain {mode}");
        }
        assert_eq!(all.iter().count(), 5);
    }

    #[test]
    fn test_enabled_permissions_from_modes() {
        let single = EnabledPermissions::from_modes([Permission::Read]).unwrap();
        assert!(single.is_enabled(Permission::Read));
        assert_eq!(single.iter().count(), 1);

        let dups = EnabledPermissions::from_modes([
            Permission::Read,
            Permission::Read,
            Permission::Unrestricted,
        ])
        .unwrap();
        assert_eq!(dups.iter().count(), 2);
        assert!(dups.is_enabled(Permission::Read));
        assert!(dups.is_enabled(Permission::Unrestricted));

        assert_eq!(EnabledPermissions::from_modes(std::iter::empty()), None);
    }

    #[test]
    fn test_enabled_permissions_iter_order() {
        let all = EnabledPermissions::ALL;
        let order: Vec<Permission> = all.iter().collect();
        assert_eq!(order, EVERY.to_vec());
    }

    #[test]
    fn test_enabled_permissions_lowest() {
        assert_eq!(EnabledPermissions::ALL.lowest(), Permission::None);
        assert_eq!(EnabledPermissions::DEFAULT.lowest(), Permission::None);
        assert_eq!(
            EnabledPermissions::from_modes([Permission::Ask, Permission::Unrestricted])
                .unwrap()
                .lowest(),
            Permission::Ask
        );
        assert_eq!(
            EnabledPermissions::from_modes([Permission::Unrestricted])
                .unwrap()
                .lowest(),
            Permission::Unrestricted
        );
    }

    #[test]
    fn test_shared_permission_basic() {
        let shared = SharedPermission::new(Permission::Read, EnabledPermissions::ALL);
        assert_eq!(shared.get(), Permission::Read);

        shared.try_set(Permission::Unrestricted).unwrap();
        assert_eq!(shared.get(), Permission::Unrestricted);
    }

    #[test]
    fn test_shared_permission_clone() {
        let shared = SharedPermission::new(Permission::Read, EnabledPermissions::ALL);
        let cloned = shared.clone();

        shared.try_set(Permission::Unrestricted).unwrap();
        assert_eq!(cloned.get(), Permission::Unrestricted);
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
        // DEFAULT is none/read/workspace/unrestricted, so Read → Workspace, then Ask is skipped.
        assert_eq!(shared.cycle(), Permission::Workspace);
        assert_eq!(shared.get(), Permission::Workspace);
        assert_eq!(shared.cycle(), Permission::Unrestricted);
        assert_eq!(shared.cycle(), Permission::None);
        assert_eq!(shared.cycle(), Permission::Read);
    }

    /// Shift+Tab reaches the confined rung before the unbounded one.
    ///
    /// Ordering, not just membership: a user cycling toward "let the agent write" stops at
    /// `workspace` first, and has to press again to give up the boundary.
    #[test]
    fn test_shared_permission_cycle_all_enabled() {
        let shared = SharedPermission::new(Permission::None, EnabledPermissions::ALL);
        assert_eq!(shared.cycle(), Permission::Read);
        assert_eq!(shared.cycle(), Permission::Workspace);
        assert_eq!(shared.cycle(), Permission::Ask);
        assert_eq!(shared.cycle(), Permission::Unrestricted);
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
