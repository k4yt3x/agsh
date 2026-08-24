//! The per-session workspace: where relative paths resolve from, and which roots a search sweeps.
//!
//! Its own module rather than a corner of `agent.rs` because every file-touching tool needs it and
//! nothing here needs an `Agent`. Living in `agent.rs` made `tools` depend on `agent` for a path
//! join, which is a cycle between the two largest modules in the tree and the reason a reader
//! looking for "how does `read_file` resolve a relative path" ended up in the turn loop.

use std::{
    path::{Path, PathBuf},
    sync::{Arc, RwLock},
};

/// Per-session working directory, shared by reference between the agent, every file-touching tool,
/// the REPL prompt, the `/cd` slash command, and the per-turn environment-context block.
/// `std::sync::RwLock` (rather than `tokio::sync::RwLock`) so the synchronous REPL prompt can read
/// it without entering an async context; reads/writes are microseconds (a `PathBuf` clone or
/// replace), never held across `.await`.
pub type SharedCwd = Arc<RwLock<PathBuf>>;

/// Read the current value of [`SharedCwd`]. Recovers from a poisoned lock by extracting the inner
/// value; meka never panics with the cwd lock held, so the only way to see a poisoned lock is a
/// separate bug that already triggered, and falling back to the stored value beats crashing the
/// agent on every subsequent tool call.
pub fn cwd_snapshot(cwd: &SharedCwd) -> PathBuf {
    cwd.read()
        .map(|guard| guard.clone())
        .unwrap_or_else(|poisoned| poisoned.into_inner().clone())
}

/// Resolve a tool-input path against the per-session [`SharedCwd`]. Absolute paths pass through
/// unchanged; relative paths are joined to the current cwd value. Tools use this at the top of
/// their `execute` methods to decouple from process `cwd`.
pub fn resolve_against_cwd(cwd: &SharedCwd, input: impl AsRef<std::path::Path>) -> PathBuf {
    let input = input.as_ref();
    if input.is_absolute() {
        input.to_path_buf()
    } else {
        cwd_snapshot(cwd).join(input)
    }
}

/// Workspace roots beyond [`SharedCwd`]: an ACP client's `additionalDirectories`, and each
/// `--writable-root` on the command line.
///
/// A separate handle rather than a field on `SharedCwd` because only the search tools, the
/// environment-context block and the write boundary care: widening [`resolve_against_cwd`] would
/// touch every file tool's constructor to serve them. `cwd` remains the base for relative paths,
/// per the ACP spec.
///
/// These expand discovery scope **and**, at [`crate::permission::Permission::Workspace`], the set
/// of roots a write may land under ([`writable_roots`]). A client that hands meka a folder is
/// naming part of the workspace rather than somewhere to search, and a mode that could read those
/// folders but never write them would impose a boundary the client never asked for.
///
/// Empty unless one of those named a root, which is the common case for the HTTP API and for an ACP
/// client that sends none.
pub type SharedRoots = Arc<RwLock<Vec<PathBuf>>>;

/// Read the current value of [`SharedRoots`], with the same poisoned-lock recovery as
/// [`cwd_snapshot`] and for the same reason.
pub fn roots_snapshot(roots: &SharedRoots) -> Vec<PathBuf> {
    roots
        .read()
        .map(|guard| guard.clone())
        .unwrap_or_else(|poisoned| poisoned.into_inner().clone())
}

/// The ordered set of roots a **recursive** search should sweep when the caller named no explicit
/// path: `cwd` first, then each additional root, with anything already covered by another root
/// dropped.
///
/// Only correct for a walker that descends, which today means `search_contents`. A tool that
/// anchors a pattern at each root instead wants [`glob_roots`]; dropping a contained root would
/// drop the files under it.
///
/// A root is dropped when some other root *contains* it, which subsumes exact duplicates. Both
/// shapes are things a client legitimately sends: Zed may repeat `cwd` inside
/// `additionalDirectories`, and nothing stops a client naming a folder nested inside another. Left
/// in, the overlapping tree is walked twice, so every file under it is reported twice, consumes two
/// slots of the result cap, and spends the shared walk budget twice.
///
/// Containment is checked in both directions, so a root that is an *ancestor* of `cwd` wins and
/// `cwd` drops out of the search set. A descending walk from the ancestor still reaches everything
/// under `cwd`, and this does not affect `cwd`'s real job: it remains the base for relative paths
/// and the shell's working directory regardless of what this returns.
///
/// Paths are compared as given. A symlink pointing at another root, or a path containing `..`, is
/// not detected; canonicalising to catch those would resolve symlinked roots to targets the client
/// never named, which is a worse trade than an occasional duplicate.
pub fn search_roots(cwd: &SharedCwd, roots: &SharedRoots) -> Vec<PathBuf> {
    retain_broadest(std::iter::once(cwd_snapshot(cwd)).chain(roots_snapshot(roots)))
}

/// Keep only the broadest roots, in first-seen order: a path contained by one already kept is
/// dropped, and a path that *contains* ones already kept replaces them.
///
/// Shared by [`search_roots`] and [`writable_roots`] because both answer questions where a
/// contained root is genuinely redundant. A descending walk from the ancestor reaches everything
/// beneath it, and a write permitted under the ancestor is permitted under its children, so in both
/// cases keeping the narrower path would only duplicate work. [`glob_roots`] deliberately does not
/// use this; see its doc comment for the failure that caused.
fn retain_broadest(paths: impl IntoIterator<Item = PathBuf>) -> Vec<PathBuf> {
    let mut kept: Vec<PathBuf> = Vec::new();
    for path in paths {
        if kept.iter().any(|existing| path.starts_with(existing)) {
            continue;
        }
        // This root is broader than ones already kept, so those become redundant.
        kept.retain(|existing| !existing.starts_with(&path));
        kept.push(path);
    }
    kept
}

/// The roots a write may land under at [`crate::permission::Permission::Workspace`]: the working
/// directory plus [`SharedRoots`].
///
/// All three sources the user can name feed [`SharedRoots`] rather than travelling separately: an
/// ACP client's `additionalDirectories`, and each `--writable-root` on the command line. They mean
/// the same thing (this folder is part of my workspace) and get the same treatment, searchable and
/// writable, so there is no second list to keep in step with this one.
///
/// **This is the one definition of the boundary.** The in-process fence on `write_file` /
/// `edit_file` / `scratchpad_save_file` and every sandbox dialect (Landlock, Bubblewrap, Seatbelt,
/// the Windows restricted token) all derive their allow-list from here and nowhere else, so the
/// file tools and the shell cannot end up disagreeing about where a write may land. That asymmetry
/// is the specific failure this function exists to make unrepresentable.
///
/// Roots come back **canonical**, with symlinks resolved. That is what makes containment checks
/// meaningful: the target of a write is canonicalised too, so a symlink planted inside the
/// workspace and pointing out of it (`<root>/escape -> /etc`) resolves to `/etc/...` and fails the
/// prefix test. Comparing as-spelled would let exactly that through. It also means the boundary is
/// stated in the filesystem's own terms rather than the user's spelling, which is what the kernel
/// backends match on.
///
/// A root that does not resolve is **dropped**, not passed through. A path that cannot be
/// canonicalised does not exist yet, and a boundary naming a directory that is not there should
/// permit nothing rather than permit a name that something could later be created at.
///
/// Empty is a meaningful answer and means "no write may land anywhere", which is what every caller
/// must do with it. It happens when the cwd has been deleted out from under a running session.
///
/// Synchronous because both kinds of caller need it: the async fence and the `pre_exec` sandbox
/// setup, where an `.await` is not available. The cost is a handful of `stat` calls per write.
pub fn writable_roots(cwd: &SharedCwd, roots: &SharedRoots) -> Vec<PathBuf> {
    usable_roots(std::iter::once(cwd_snapshot(cwd)).chain(roots_snapshot(roots)))
}

/// The boundary [`writable_roots`] would compute from an already-taken snapshot of the same paths.
///
/// Exists so the `[Environment context]` block can name the roots that will actually hold, rather
/// than the roots the session was *asked* for. The two differ: a root that no longer resolves, one
/// that is a file, one that is a masked system directory, and one already contained by another are
/// each dropped here. Listing the request told the model it could write to paths where the very
/// next write would be refused, and hid the merge when two roots collapsed into one.
pub fn usable_roots(paths: impl IntoIterator<Item = PathBuf>) -> Vec<PathBuf> {
    retain_broadest(
        paths
            .into_iter()
            .filter_map(|path| std::fs::canonicalize(&path).ok().map(strip_verbatim))
            .filter(|path| is_usable_root(path)),
    )
}

/// Whether a canonical path can serve as a workspace root at all.
///
/// Two refusals, each because a backend cannot express the boundary otherwise.
///
/// **Not a directory.** Landlock's `PATH_BENEATH` rule is rejected with `EINVAL` when the parent is
/// a regular file and the rule carries directory-class rights, which meka's full handled-access
/// mask does. `apply_landlock` then fails inside `pre_exec`, so *every* shell command in the
/// session dies with `Invalid argument` -- not just a write to that root. The other three backends
/// accept a file root and quietly do something different with it, so this is also the sharpest
/// cross-backend divergence there is. `--writable-root ./notes.md` was enough to trigger it.
///
/// **A system directory that Bubblewrap masks.** The bwrap backend binds each root *after* its
/// tmpfs masks so a root under `/tmp` survives, and later mounts win -- which means a root at or
/// above a masked path un-masks it. Those masks are not about the filesystem: they exist to put the
/// D-Bus and systemd-user sockets out of reach. `--writable-root /run/user/1000` therefore hands
/// back the session bus, and `systemd-run --user` writes anywhere; a root of `/` additionally
/// un-masks `/proc` and `/dev`, defeating the PID namespace. None of these are workspaces, and
/// every backend is degraded by them, so they are refused rather than special-cased per backend.
fn is_usable_root(path: &Path) -> bool {
    if !path.is_dir() {
        tracing::warn!(
            "workspace root {} is not a directory; ignoring it. A root has to be a directory: \
             Landlock refuses a file-backed rule outright and would fail every shell command in \
             the session",
            path.display()
        );
        return false;
    }
    if is_system_root(path) {
        tracing::warn!(
            "refusing {} as a workspace root: it is a system directory whose contents the sandbox \
             masks to keep IPC sockets out of reach, and binding it back would undo that. Name the \
             project directory you actually want to write in",
            path.display()
        );
        return false;
    }
    true
}

/// Paths that must never become workspace roots. Compared against the canonical form.
///
/// A root is refused when it *is* one of these, and when it is an **ancestor** of one. Both
/// directions un-mask: bwrap binds each root after its tmpfs masks and the later mount wins, so
/// `--writable-root /var` restores the host's world-writable `/var/tmp` inside the sandbox just as
/// surely as `--writable-root /var/tmp` would. Measured both ways. The ancestor case is the one
/// that bites in practice, because `$XDG_RUNTIME_DIR` lives under `$HOME` on WSL, on minimal
/// window managers, and wherever someone set `XDG_RUNTIME_DIR=$HOME/.xdg`; a root at `$HOME` then
/// hands the session bus socket back to a confined shell, which is exactly what the masks exist to
/// prevent.
///
/// A root *under* a masked path is fine and is deliberately allowed -- the bind restores only that
/// subdirectory, not the masked directory itself -- except for the two socket trees below, which
/// are refused as whole subtrees because everything in them is the kind of socket being hidden.
///
/// An earlier version of this comment said `/tmp` and `/var/tmp` were "deliberately absent" from
/// the set because they "hold no IPC socket meka's masks care about". That was false on any
/// ordinary desktop and is recorded in the changelog as a security fix; the list below is the
/// authority.
pub(crate) fn is_system_root(path: &Path) -> bool {
    if path.parent().is_none() {
        // The filesystem root itself, and on Windows a bare drive prefix.
        return true;
    }
    #[cfg(unix)]
    {
        // A root **at** one of these is refused; a root **under** one is fine, and the difference
        // is the whole point.
        //
        // Bubblewrap masks each of these with a tmpfs and then binds the workspace back afterwards,
        // last-mount-wins. Binding `/tmp/work` restores exactly the workspace. Binding `/tmp`
        // restores the entire host `/tmp`, including every X11, D-Bus and tmux socket living there,
        // which is a hole straight back out of the sandbox: measured, by reaching a tmux server
        // over a socket under `/tmp` from inside a confined shell and having it create a file in
        // `$HOME`, outside every workspace root. `/tmp` and `/var/tmp` used to be admitted on the
        // reasoning that they "hold no IPC socket meka's masks care about", which is false on any
        // ordinary desktop.
        //
        // Refusing `/tmp` costs a session started with `cd /tmp` its write boundary, which is the
        // safe direction: the alternative is a boundary that reports itself as holding while the
        // shell can reach the session bus.
        const MASKED: &[&str] = &["/proc", "/dev", "/sys", "/run", "/tmp", "/var/tmp"];
        // Equal to a masked path, or an ancestor of one. The equality test alone left the ancestor
        // case open: `--writable-root /var` was accepted and then un-masked `/var/tmp` inside the
        // sandbox, measured by writing a file there from a confined shell and finding it on the
        // host. `starts_with` is component-wise, so `/vary` is not an ancestor of `/var/tmp`.
        if MASKED
            .iter()
            .map(Path::new)
            .any(|masked| path == masked || masked.starts_with(path))
        {
            return true;
        }
        // The session's socket tree, refused as a *subtree* because everything in it is the kind of
        // socket the masks exist to hide. Checked by path as well as by variable so an unset or
        // stale `XDG_RUNTIME_DIR` does not open it.
        if path.starts_with("/run/user") || Path::new("/run/user").starts_with(path) {
            return true;
        }
        if let Some(runtime) = std::env::var_os("XDG_RUNTIME_DIR")
            && !runtime.is_empty()
            && let Ok(runtime) = std::fs::canonicalize(&runtime)
            // Both directions, and the ancestor half is the one that matters: this directory is
            // often under `$HOME`, so a root at `$HOME` -- or the far more ordinary `cd ~` -- put
            // the session bus back within reach of a confined shell. Measured by connecting to a
            // socket under a simulated `$XDG_RUNTIME_DIR` from inside bwrap with an ancestor root.
            && (path.starts_with(&runtime) || runtime.starts_with(path))
        {
            return true;
        }
    }
    false
}

/// Serialises the tests that point `XDG_RUNTIME_DIR` at a directory of their own.
///
/// Same shape and same reason as [`crate::config::CONFIG_DIR_ENV_LOCK`]: the variable is
/// process-global, `cargo test` runs these in parallel, and [`is_system_root`] reads it.
#[cfg(all(test, unix))]
pub(crate) static RUNTIME_DIR_ENV_LOCK: std::sync::Mutex<()> = std::sync::Mutex::new(());

/// Drop Windows' `\\?\` verbatim prefix, which only `canonicalize` speaks.
///
/// Invisible on Unix and load-bearing on Windows. `canonicalize` returns `\\?\C:\ws` while a cwd
/// from `current_dir` and a path the model wrote are both spelled `C:\ws`, and the former never
/// prefix-matches the latter. The fence therefore refused every write *inside* the workspace while
/// still refusing the ones outside it, which reads as a working boundary from every angle except
/// the one that matters. Only a live Windows turn surfaced it.
///
/// Verbatim UNC paths are deliberately left alone: a network share cannot be a workspace root
/// anyway, because meka has to own the directory to grant on it.
///
/// `pub(crate)` because `/cd` needs it too: it canonicalises what the user typed and stores the
/// result as the session working directory, from which the prompt, the model's environment block,
/// every relative tool path and the sessions table's `cwd` column are all derived. This is the
/// tree's one answer to what shape a path is in, so everything that produces one goes through it.
pub(crate) fn strip_verbatim(path: PathBuf) -> PathBuf {
    #[cfg(windows)]
    {
        use std::path::{Component, Prefix};
        let mut components = path.components();
        if let Some(Component::Prefix(prefix)) = components.next()
            && let Prefix::VerbatimDisk(letter) = prefix.kind()
        {
            // Rebuilt from the remaining components rather than by trimming the rendered string. A
            // path is a sequence of `OsStr`s, not text: `to_string_lossy` would replace any
            // unpaired surrogate in a Windows filename with U+FFFD, and comparing a boundary
            // against a path that has silently changed is exactly the class of bug this function
            // exists to fix. `VerbatimDisk` carries an ASCII drive letter by construction, so the
            // designator is the one part that can safely be built as text.
            let mut rebuilt = PathBuf::from(format!("{}:", letter as char));
            rebuilt.extend(components);
            return rebuilt;
        }
    }
    path
}

/// Whether `path` lies within one of `roots`.
///
/// Split out so the fence and its tests agree on what containment means, and so a root that equals
/// the path is a hit: writing *to* a workspace root's own path is a write inside it.
///
/// Both sides are put in the same normal form here rather than at each call site. Callers arrive
/// from two directions, `canonicalize` (verbatim on Windows) and lexical normalisation of a path
/// that does not exist yet (never verbatim), and a comparison that assumed either one would be
/// wrong half the time.
pub fn is_within_roots(path: &std::path::Path, roots: &[PathBuf]) -> bool {
    let path = strip_verbatim(path.to_path_buf());
    roots
        .iter()
        .any(|root| path.starts_with(strip_verbatim(root.clone())))
}

/// Normalise `.` and `..` textually, without consulting the filesystem.
///
/// The filesystem cannot help for the case this exists for: a `write_file` naming a path that does
/// not exist yet, which is most of them. `canonicalize` fails on a missing path, so the only way to
/// judge `<root>/../../etc/passwd` *before* creating anything is to resolve the components as
/// text. It is not a substitute for canonicalisation, which still runs afterwards to catch the
/// symlinked ancestor this pass cannot see.
fn normalize_lexically(path: &std::path::Path) -> PathBuf {
    let mut out = PathBuf::new();
    for component in path.components() {
        match component {
            std::path::Component::ParentDir => {
                out.pop();
            }
            std::path::Component::CurDir => {}
            other => out.push(other.as_os_str()),
        }
    }
    out
}

/// The write boundary as a tool sees it: whether one applies right now, and what it admits.
///
/// Carries the *live* permission handle rather than a snapshot, so a `/permission` change mid-turn
/// reaches the next write rather than the next session. Cheap to clone; every field is shared.
#[derive(Clone)]
pub struct WriteScope {
    permission: crate::permission::SharedPermission,
    roots: SharedRoots,
    /// Set only by [`WriteScope::deny_all`]. A separate flag rather than an empty root list,
    /// because an empty list does not mean "nothing": [`writable_roots`] always folds in the cwd,
    /// so the closed fallback was in fact a cwd-wide grant. There has to be a state that means
    /// *no* root, and the root list cannot express it.
    denied: bool,
}

impl WriteScope {
    pub fn new(permission: crate::permission::SharedPermission, roots: SharedRoots) -> Self {
        Self {
            permission,
            roots,
            denied: false,
        }
    }

    /// The roots a write must land under right now, or `None` when this level imposes no boundary.
    ///
    /// Only `workspace` confines. `ask` deliberately does not: its safety is the approval prompt,
    /// and an approved call is meant to reach anywhere. `unrestricted` does not by definition, and
    /// the levels below it never reach a write door at all.
    pub fn confined_to(&self, cwd: &SharedCwd) -> Option<Vec<PathBuf>> {
        if self.denied {
            return Some(Vec::new());
        }
        match self.permission.get() {
            // Only the two levels that *disclaim* a boundary are exempt. Written as an allow-list
            // rather than `Workspace => Some(..), _ => None`, because that catch-all failed open:
            // it exempted `none` and `read` too, on the reasoning that they never reach a write
            // door. They normally do not -- but `[tools.tool_permissions]` overrides a tool's
            // required level with no floor, so `write_file = "read"` dispatches the tool at `read`
            // and the fence then waved it through with no boundary at all.
            //
            // At `none` and `read` this yields the workspace roots rather than nothing, which is a
            // deliberate difference from `Confinement::resolve`, whose catch-all is `ReadOnly` and
            // grants no write at all. The two are *not* the same decision, and an earlier version
            // of this comment claimed they were:
            //
            // - `Confinement::resolve` answers "what may a command meka did not write do", and the
            //   honest answer at `read` is nothing.
            // - This answers "where may a tool the operator deliberately lowered write", and a
            //   `write_file = "read"` override is a statement that the tool should be usable at
            //   that level. Refusing it outright would make the override a no-op with no
            //   diagnostic; confining it to the workspace roots is the narrowest reading that still
            //   honours what was configured.
            //
            // So the override can only ever narrow *reach*: it never escapes the roots, and it
            // cannot touch `execute_command`, which stays closed at `read` regardless.
            crate::permission::Permission::Ask | crate::permission::Permission::Unrestricted => {
                None
            }
            _ => Some(writable_roots(cwd, &self.roots)),
        }
    }

    /// Judge one write target, returning the refusal sentence when it falls outside the boundary.
    ///
    /// `target` should be canonical where the caller can manage it and lexically normalised where
    /// it cannot (a path being created). Both are checked the same way; the difference is only in
    /// how much the caller has been able to resolve.
    pub fn admit(&self, cwd: &SharedCwd, target: &std::path::Path) -> Result<(), String> {
        let Some(roots) = self.confined_to(cwd) else {
            return Ok(());
        };
        let candidate = normalize_lexically(target);
        if is_within_roots(&candidate, &roots) {
            return Ok(());
        }
        // Names the roots rather than just refusing: the model cannot see the boundary from the
        // tool schema, and without them its next attempt is a guess. Empty is a real state (the
        // working directory was deleted), and saying so beats reporting an empty list.
        let where_to = if roots.is_empty() {
            "no workspace root currently resolves, so no write can land anywhere".to_string()
        } else {
            format!(
                "writes must land under {}",
                roots
                    .iter()
                    .map(|root| root.display().to_string())
                    .collect::<Vec<_>>()
                    .join(", ")
            )
        };
        Err(format!(
            "'{}' is outside the workspace: at `workspace` permission {}. Pick a path inside, or \
             ask the user for `unrestricted` if it genuinely belongs elsewhere.",
            target.display(),
            where_to
        ))
    }

    /// A scope that admits nothing, for a registry that never registered the core tools and so
    /// holds no permission handle to judge against.
    ///
    /// Unreachable in production: both `build_default` and `build_for_subagent` call
    /// `register_core_tools` before the session-scoped pass. It exists so the fallback is
    /// *closed*. A registry that cannot tell whether a boundary applies must refuse rather than
    /// assume there is none, which is the direction a missing scope would otherwise fail in.
    pub fn deny_all() -> Self {
        Self {
            permission: crate::permission::SharedPermission::new(
                crate::permission::Permission::None,
                crate::permission::EnabledPermissions::DEFAULT,
            ),
            roots: Arc::new(RwLock::new(Vec::new())),
            denied: true,
        }
    }

    /// A scope that confines nothing, for tools under test that are not exercising the boundary.
    #[cfg(test)]
    pub fn unconfined() -> Self {
        Self::new(
            crate::permission::SharedPermission::new(
                crate::permission::Permission::Unrestricted,
                crate::permission::EnabledPermissions::ALL,
            ),
            test_roots(),
        )
    }

    /// A scope confined to `roots`, for tests that *are* exercising the boundary.
    #[cfg(test)]
    pub fn confined(roots: Vec<PathBuf>) -> Self {
        Self::new(
            crate::permission::SharedPermission::new(
                crate::permission::Permission::Workspace,
                crate::permission::EnabledPermissions::ALL,
            ),
            Arc::new(RwLock::new(roots)),
        )
    }
}

/// The ordered set of roots to anchor a glob at when the caller named no explicit path: `cwd`
/// first, then each additional root, with only *exact* repeats dropped.
///
/// The counterpart to [`search_roots`] for a tool that builds one rooted pattern per root rather
/// than descending from it. Containment must not drop anything here: `find_files` turns each root
/// into `<root>/<pattern>`, and a glob's `*` does not cross `/`, so a workspace of `/work` plus
/// `cwd = /work/main` would answer `*.md` from `/work/*.md` alone and miss
/// `/work/main/README.md` entirely. That is the exact "the agent says a file you can see doesn't
/// exist" failure multi-root support was added to prevent, so nested roots are all kept and the
/// caller deduplicates the matches instead.
pub fn glob_roots(cwd: &SharedCwd, roots: &SharedRoots) -> Vec<PathBuf> {
    let mut kept: Vec<PathBuf> = Vec::new();
    for path in std::iter::once(cwd_snapshot(cwd)).chain(roots_snapshot(roots)) {
        if !kept.contains(&path) {
            kept.push(path);
        }
    }
    kept
}

/// Canonicalise a path the way meka does, for tests that compare against meka's own output.
///
/// `std::fs::canonicalize` hands back a `\\?\`-prefixed path on Windows, and every production
/// caller runs the result through [`strip_verbatim`]. A test that skips that step is asserting
/// against the one spelling meka never produces -- and it passes everywhere `strip_verbatim` is the
/// identity, so the failure is Windows-only and invisible until something runs there. Five tests
/// reached the tree with that shape at once, which is why this is a named helper rather than a
/// `.map(strip_verbatim)` a future test can forget.
#[cfg(test)]
pub fn canonical_for_test(path: impl AsRef<std::path::Path>) -> PathBuf {
    std::fs::canonicalize(path.as_ref())
        .map(strip_verbatim)
        .unwrap_or_else(|error| panic!("canonicalize {}: {error}", path.as_ref().display()))
}

/// Construct a fresh [`SharedCwd`] pointing at the process cwd, for use in tests that need to
/// instantiate a tool but don't exercise the per-session cwd resolution path. Tests using absolute
/// paths or `tempdir()` are unaffected by the value here.
#[cfg(test)]
pub fn test_cwd() -> SharedCwd {
    Arc::new(RwLock::new(
        std::env::current_dir().unwrap_or_else(|_| PathBuf::from(".")),
    ))
}

/// Construct an empty [`SharedRoots`], for tools under test that don't exercise multi-root search.
#[cfg(test)]
pub fn test_roots() -> SharedRoots {
    Arc::new(RwLock::new(Vec::new()))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shared(path: &std::path::Path) -> SharedCwd {
        Arc::new(RwLock::new(path.to_path_buf()))
    }

    /// The three sources compose into one canonical, containment-deduplicated set.
    #[test]
    fn writable_roots_combines_the_cwd_with_every_named_root() {
        let temp = tempfile::tempdir().expect("tempdir");
        let base = crate::workspace::canonical_for_test(temp.path());
        for name in ["work", "shared", "extra"] {
            std::fs::create_dir(base.join(name)).expect("create dir");
        }

        let roots = writable_roots(
            &shared(&base.join("work")),
            &Arc::new(RwLock::new(vec![base.join("shared"), base.join("extra")])),
        );

        assert_eq!(roots, vec![
            base.join("work"),
            base.join("shared"),
            base.join("extra"),
        ]);
    }

    /// Containment is judged on the *resolved* path, so a link that leads out of the workspace is
    /// not inside it once resolved.
    ///
    /// Named for what it actually checks. `is_within_roots` is a component-wise prefix test and
    /// deliberately does no symlink resolution of its own -- `normalize_lexically` is documented as
    /// not touching them -- so the resolution this depends on happens in the caller
    /// (`resolve_write_target`, via `resolve_existing_prefix`). The fixture stands in for that
    /// caller by canonicalising before it asks.
    ///
    /// The test used to be named as though this layer resolved the link, which is false of the code
    /// under test: feeding it the *spelled* path would return `true`, and the reason that is safe
    /// is that no caller ever does.
    #[test]
    #[cfg(unix)]
    fn a_resolved_path_leading_out_of_the_workspace_is_not_within_it() {
        let temp = tempfile::tempdir().expect("tempdir");
        let base = crate::workspace::canonical_for_test(temp.path());
        let work = base.join("work");
        let outside = base.join("outside");
        std::fs::create_dir(&work).expect("work");
        std::fs::create_dir(&outside).expect("outside");
        std::os::unix::fs::symlink(&outside, work.join("escape")).expect("symlink");

        let roots = writable_roots(&shared(&work), &test_roots());
        let spelled = work.join("escape").join("passwd");
        let escaped = crate::workspace::canonical_for_test(work.join("escape")).join("passwd");

        assert!(is_within_roots(&work.join("src"), &roots));
        // The spelled form *is* admitted, and stating that is the point: it is what makes
        // resolving-before-asking load-bearing rather than decorative. An implementation that
        // compared as written would take this path and land the bytes in `outside`.
        assert!(
            is_within_roots(&spelled, &roots),
            "this layer does not resolve links, so the spelled path passes -- which is exactly why \
             the caller must resolve first"
        );
        assert!(
            !is_within_roots(&escaped, &roots),
            "a link resolving outside the root must not be inside it"
        );
    }

    /// The boundary tracks the cwd rather than freezing at the value it was built with.
    ///
    /// Documented behaviour, not an accident of the implementation: `/cd` is meant to move the
    /// workspace, and a boundary that stayed put would refuse writes to the directory the user is
    /// plainly now working in. Named roots are unaffected by the move, which is what keeps a
    /// client-supplied folder the client's rather than the cwd's.
    #[test]
    fn writable_roots_follow_the_cwd_when_it_moves() {
        let temp = tempfile::tempdir().expect("tempdir");
        let base = crate::workspace::canonical_for_test(temp.path());
        for name in ["before", "after", "named"] {
            std::fs::create_dir(base.join(name)).expect("create dir");
        }
        let cwd = shared(&base.join("before"));
        let named: SharedRoots = Arc::new(RwLock::new(vec![base.join("named")]));

        assert_eq!(writable_roots(&cwd, &named), vec![
            base.join("before"),
            base.join("named"),
        ]);

        match cwd.write() {
            Ok(mut current) => *current = base.join("after"),
            Err(poisoned) => *poisoned.into_inner() = base.join("after"),
        }

        assert_eq!(
            writable_roots(&cwd, &named),
            vec![base.join("after"), base.join("named")],
            "the boundary must move with the cwd and leave named roots alone"
        );
    }

    /// A canonical root and an as-spelled target inside it agree, despite Windows' verbatim prefix.
    ///
    /// The regression this guards shipped once and was invisible to every Linux test:
    /// `canonicalize` hands back `\\?\C:\ws`, the fence checks a target spelled `C:\ws\f.txt`
    /// against it, and `starts_with` says no. Writes *outside* were still refused, so the
    /// boundary looked correct while refusing everything the mode exists to allow.
    #[test]
    #[cfg(windows)]
    fn a_verbatim_root_admits_an_as_spelled_path_inside_it() {
        use std::path::{Component, Prefix};

        let temp = tempfile::tempdir().expect("tempdir");
        // Deliberately *not* `canonical_for_test`: this test is about the verbatim prefix, so it
        // needs the raw spelling the helper exists to remove.
        let canonical = std::fs::canonicalize(temp.path()).expect("canonicalize");
        assert!(
            matches!(
                canonical.components().next(),
                Some(Component::Prefix(prefix)) if matches!(prefix.kind(), Prefix::VerbatimDisk(_))
            ),
            "precondition: Windows canonicalize returns a verbatim path, got {}",
            canonical.display()
        );
        // The rebuild must preserve every component, not merely drop the prefix.
        assert_eq!(
            strip_verbatim(canonical.join("a").join("b")),
            temp.path().join("a").join("b")
        );

        let roots = writable_roots(&shared(temp.path()), &test_roots());
        assert!(
            is_within_roots(&temp.path().join("f.txt"), &roots),
            "an as-spelled path inside the root must be admitted: roots {roots:?}"
        );
        assert!(
            is_within_roots(&canonical.join("f.txt"), &roots),
            "a verbatim path inside the root must be admitted too: roots {roots:?}"
        );
        assert!(!is_within_roots(
            std::path::Path::new(r"C:\Windows\x"),
            &roots
        ));
    }

    /// A file is not a workspace root, and neither is a masked system directory.
    ///
    /// Both refusals exist because a backend cannot express the boundary otherwise, and both were
    /// reachable from `--writable-root` or an ACP client. A file root makes Landlock reject its own
    /// rule with `EINVAL` inside `pre_exec`, which kills *every* shell command in the session
    /// rather than just a write to that root. A root at or above one of Bubblewrap's tmpfs masks
    /// un-masks it, and those masks are what keep the D-Bus and systemd sockets out of reach -- so
    /// `--writable-root /run/user/1000` turned a workspace grant into `systemd-run --user`.
    #[test]
    fn writable_roots_refuses_a_file_and_a_masked_system_directory() {
        let temp = tempfile::tempdir().expect("tempdir");
        let base = crate::workspace::canonical_for_test(temp.path());
        let work = base.join("work");
        std::fs::create_dir(&work).expect("work");
        let not_a_directory = base.join("notes.md");
        std::fs::write(&not_a_directory, b"x").expect("write");

        let roots = writable_roots(
            &shared(&work),
            &Arc::new(RwLock::new(vec![not_a_directory.clone()])),
        );
        assert_eq!(roots, vec![work.clone()], "a file must not become a root");

        #[cfg(unix)]
        {
            let roots = writable_roots(
                &shared(&work),
                &Arc::new(RwLock::new(vec![
                    PathBuf::from("/run"),
                    PathBuf::from("/proc"),
                    PathBuf::from("/"),
                ])),
            );
            assert_eq!(
                roots,
                vec![work],
                "no masked system directory may become a root: {roots:?}"
            );
        }
    }

    /// A runtime directory named only by `$XDG_RUNTIME_DIR` is refused in both directions.
    ///
    /// The whole environment branch had no test: deleting the `!` from its `is_empty` guard, which
    /// disables it outright, left the suite green. What covered the socket tree was the `/run/user`
    /// literal beside it, and only because the developer's own runtime directory happens to live
    /// there.
    ///
    /// The ancestor direction is the half that matters. `$XDG_RUNTIME_DIR` sits under `$HOME` on
    /// WSL and on minimal window managers, so an ordinary `cd ~` names an ancestor of the session
    /// bus, and a boundary that only refused the directory itself would hand the bus back to a
    /// confined shell.
    #[test]
    #[cfg(unix)]
    fn a_runtime_directory_named_by_the_environment_is_refused_from_either_side() {
        let temp = tempfile::tempdir().expect("tempdir");
        let base = crate::workspace::canonical_for_test(temp.path());
        let runtime = base.join("runtime");
        let socket = runtime.join("bus");
        std::fs::create_dir_all(&socket).expect("dirs");
        let sibling = base.join("project");
        std::fs::create_dir(&sibling).expect("sibling");

        let _guard = RUNTIME_DIR_ENV_LOCK
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        let previous = std::env::var_os("XDG_RUNTIME_DIR");
        // SAFETY: `XDG_RUNTIME_DIR` is process-global; `RUNTIME_DIR_ENV_LOCK` serialises every test
        // that touches it and the guard is held across the whole set/read/restore cycle.
        unsafe { std::env::set_var("XDG_RUNTIME_DIR", &runtime) };

        let at = is_system_root(&runtime);
        let under = is_system_root(&socket);
        let above = is_system_root(&base);
        let beside = is_system_root(&sibling);

        // Restored before asserting: a panic here must not leak the variable into every test that
        // runs after it.
        match previous {
            Some(value) => unsafe { std::env::set_var("XDG_RUNTIME_DIR", value) },
            None => unsafe { std::env::remove_var("XDG_RUNTIME_DIR") },
        }

        assert!(at, "the runtime directory itself must be refused");
        assert!(under, "a socket inside it must be refused");
        assert!(
            above,
            "an ancestor un-masks the whole tree, which is what `cd ~` does on WSL"
        );
        assert!(
            !beside,
            "a sibling holds no session socket and must stay usable as a workspace root"
        );
    }

    /// A root *above* a masked directory un-masks it just as surely as a root *at* it.
    ///
    /// bwrap binds each root after its tmpfs masks and the later mount wins, so `--writable-root
    /// /var` restores the host's world-writable `/var/tmp` inside the sandbox. Measured: a confined
    /// shell with that root wrote a file to `/var/tmp` and it appeared on the host, where the same
    /// write with a root elsewhere landed in the ephemeral tmpfs.
    ///
    /// The doc on `is_usable_root` said "at or above" from the start; only "at" was implemented,
    /// and the gap is reachable without any exotic setup -- `$XDG_RUNTIME_DIR` sits under `$HOME`
    /// on WSL and on minimal window managers, so an ordinary `cd ~` handed the session bus back.
    #[test]
    #[cfg(unix)]
    fn a_root_above_a_masked_directory_is_refused_too() {
        for ancestor in ["/var", "/run"] {
            assert!(
                is_system_root(std::path::Path::new(ancestor)),
                "{ancestor} is an ancestor of a masked directory and must be refused"
            );
        }
        // The control: an ordinary directory that merely shares a textual prefix with one.
        assert!(
            !is_system_root(std::path::Path::new("/vary")),
            "`starts_with` is component-wise, so /vary is not an ancestor of /var/tmp"
        );
    }

    /// A root **at** a masked directory is refused; a root **under** one is not.
    ///
    /// Both halves are load-bearing and they pull in opposite directions, which is why they are
    /// asserted together. Admitting `/tmp` itself is a sandbox escape: bubblewrap masks it with a
    /// tmpfs and then binds each workspace root back afterwards, so `--bind-try /tmp /tmp` restores
    /// the whole host `/tmp` and with it every X11, D-Bus and tmux socket living there. Refusing
    /// everything *under* a masked directory is the opposite failure: `/run/media/<user>/<disk>` is
    /// where udisks2 mounts removable drives, so a project on an external drive resolved fine and
    /// was then discarded, leaving a boundary that permitted nothing anywhere.
    #[test]
    #[cfg(unix)]
    fn a_root_at_a_masked_directory_is_refused_but_one_under_it_is_not() {
        for refused in ["/tmp", "/var/tmp", "/run", "/proc", "/dev", "/sys"] {
            assert!(
                is_system_root(Path::new(refused)),
                "{refused} is masked, so binding it back would un-mask what the mask hides"
            );
        }
        for admitted in [
            "/run/media/someone/disk/project",
            "/tmp/work",
            "/var/tmp/build",
            "/dev/shm/scratch",
        ] {
            assert!(
                !is_system_root(Path::new(admitted)),
                "{admitted} is under a masked directory, not at it, and binding it back restores \
                 only itself"
            );
        }
        // The session's socket tree stays refused as a whole, with or without the variable set,
        // because every path in it is the kind of socket the masks exist to hide. The uid is one
        // that cannot be this process's own: with the developer's real `XDG_RUNTIME_DIR` pointing
        // at `/run/user/<their uid>`, asserting on that path passed through the *environment*
        // branch below and left this literal one unguarded, which a mutation caught.
        assert!(is_system_root(Path::new("/run/user/99999")));
        assert!(is_system_root(Path::new("/run/user/99999/bus")));

        // And the real pipeline agrees: a tempdir (which lives under `/tmp` on this host) is a
        // usable root, while `/tmp` itself is not.
        let temp = tempfile::tempdir().expect("tempdir");
        let base = crate::workspace::canonical_for_test(temp.path());
        assert_eq!(usable_roots([base.clone()]), vec![base]);
        assert!(usable_roots([PathBuf::from("/tmp")]).is_empty());
    }

    /// The closed fallback actually admits nothing, including under the working directory.
    ///
    /// It did not. `deny_all` was built at `Workspace` with an empty root list, and
    /// `writable_roots` unconditionally folds in the cwd -- so the scope a registry falls back to
    /// when it cannot tell whether a boundary applies granted the entire working tree. The doc
    /// said "admits nothing" and the accessor was renamed to say `deny_all`, and both were wrong
    /// about the same object. Unreachable in production, which is exactly why nothing noticed.
    #[test]
    fn deny_all_admits_nothing_not_even_the_cwd() {
        let temp = tempfile::tempdir().expect("tempdir");
        let base = crate::workspace::canonical_for_test(temp.path());
        let cwd = shared(&base);
        let scope = WriteScope::deny_all();

        assert_eq!(
            scope.confined_to(&cwd),
            Some(Vec::new()),
            "the closed fallback must name no writable root at all"
        );
        assert!(
            scope.admit(&cwd, &base.join("f.txt")).is_err(),
            "a write in the working directory must be refused by the closed fallback"
        );
        assert!(scope.admit(&cwd, &base).is_err());
    }

    /// Which levels confine, spelled out for all five.
    ///
    /// `ask` returning `None` is a deliberate exemption and had no test, so nothing distinguished
    /// "ask is unconfined on purpose" from "ask fell through a catch-all", which is the shape the
    /// `_ => None` arm here actually had: it exempted `none` and `read` as well, and
    /// `[tools.tool_permissions]` can dispatch a write tool at either of those.
    #[test]
    fn only_ask_and_unrestricted_disclaim_a_write_boundary() {
        let temp = tempfile::tempdir().expect("tempdir");
        let base = crate::workspace::canonical_for_test(temp.path());
        let cwd = shared(&base);

        for level in [
            crate::permission::Permission::None,
            crate::permission::Permission::Read,
            crate::permission::Permission::Workspace,
            crate::permission::Permission::Ask,
            crate::permission::Permission::Unrestricted,
        ] {
            let scope = WriteScope::new(
                crate::permission::SharedPermission::new(
                    level,
                    crate::permission::EnabledPermissions::ALL,
                ),
                test_roots(),
            );
            let outside = std::env::temp_dir().join("meka-outside-every-root.txt");
            match level {
                crate::permission::Permission::Ask
                | crate::permission::Permission::Unrestricted => {
                    assert_eq!(
                        scope.confined_to(&cwd),
                        None,
                        "{level} disclaims a boundary: an approved or unrestricted write reaches \
                         anywhere"
                    );
                    scope
                        .admit(&cwd, &outside)
                        .expect("an unconfined level admits a path outside every root");
                }
                _ => {
                    assert_eq!(
                        scope.confined_to(&cwd),
                        Some(vec![base.clone()]),
                        "{level} must be confined to the working directory"
                    );
                    scope
                        .admit(&cwd, &base.join("f.txt"))
                        .expect("a write inside the workspace is admitted");
                    assert!(
                        scope.admit(&cwd, &outside).is_err(),
                        "{level} must refuse a write outside every root"
                    );
                }
            }
        }
    }

    /// A `..` that climbs out of the workspace is refused even though the path is never created.
    ///
    /// `admit` is reached with a path that does not exist yet on every `write_file` to a new file,
    /// so `canonicalize` cannot answer and the only defence is resolving the components as text
    /// first. Without that, `starts_with` is component-wise and answers *yes* for
    /// `<root>/../../etc/passwd`, because the literal path really does begin with `<root>` -- so
    /// the escape is admitted by the check written to refuse it.
    ///
    /// Both halves are asserted. The refusal alone would still pass against an implementation that
    /// refused everything containing `..`, which would break `<root>/a/../b`: a legitimate write
    /// that normalises back inside.
    #[test]
    fn a_parent_traversal_out_of_the_workspace_is_refused_before_the_file_exists() {
        let temp = tempfile::tempdir().expect("tempdir");
        let base = crate::workspace::canonical_for_test(temp.path());
        let work = base.join("work");
        std::fs::create_dir(&work).expect("work");
        let cwd = shared(&work);
        let scope = WriteScope::new(
            crate::permission::SharedPermission::new(
                crate::permission::Permission::Workspace,
                crate::permission::EnabledPermissions::ALL,
            ),
            test_roots(),
        );

        let escape = work.join("..").join("..").join("etc").join("passwd");
        assert!(
            !escape.exists(),
            "the point of this test is a target `canonicalize` cannot resolve"
        );
        let refusal = scope
            .admit(&cwd, &escape)
            .expect_err("a `..` leaving the workspace must be refused");
        assert!(
            refusal.contains("outside the workspace"),
            "the refusal must name the boundary: {refusal}"
        );

        scope
            .admit(&cwd, &work.join("a").join("..").join("b.txt"))
            .expect("a `..` that normalises back inside the workspace is an ordinary write");
    }

    /// A root that does not resolve permits nothing rather than permitting its name.
    ///
    /// Empty is the honest answer when the cwd has been deleted under a running session, and every
    /// caller has to treat it as "no write may land anywhere".
    #[test]
    fn an_unresolvable_root_is_dropped_rather_than_passed_through() {
        let temp = tempfile::tempdir().expect("tempdir");
        let base = crate::workspace::canonical_for_test(temp.path());
        let missing = base.join("was-deleted");

        assert!(writable_roots(&shared(&missing), &test_roots()).is_empty());

        // Against a *populated* root set, which is what makes this an independent check.
        //
        // It used to run against the roots this same call had just proved empty, so
        // `is_within_roots` was `.any()` over `&[]` -- false for every input, and true of the
        // function no matter what it did. Only an `is_within_roots` that always returned `true`
        // could fail it. The comment above it described a fix that had not been made.
        let real = base.join("real");
        std::fs::create_dir(&real).expect("real root");
        let roots = writable_roots(&shared(&real), &test_roots());
        assert!(!roots.is_empty(), "the control root must resolve");
        assert!(
            is_within_roots(&real.join("f.txt"), &roots),
            "a path under a resolvable root is inside it"
        );
        assert!(
            !is_within_roots(&missing.join("f.txt"), &roots),
            "a path under an unresolvable root is inside nothing, even when other roots exist"
        );
    }

    /// A nested root is redundant: anything under it is already permitted by its ancestor.
    #[test]
    fn writable_roots_drops_roots_contained_by_another() {
        let temp = tempfile::tempdir().expect("tempdir");
        let base = crate::workspace::canonical_for_test(temp.path());
        std::fs::create_dir_all(base.join("work/nested")).expect("dirs");

        let roots = writable_roots(
            &shared(&base.join("work")),
            &Arc::new(RwLock::new(vec![base.join("work/nested")])),
        );
        assert_eq!(roots, vec![base.join("work")]);
        assert!(is_within_roots(&base.join("work/nested/f.txt"), &roots));
    }

    /// `cwd` leads and duplicates are dropped: a client is free to repeat `cwd` inside
    /// `additionalDirectories`, and a repeated root would double every search result and spend the
    /// shared walk budget twice on the same tree.
    #[test]
    fn test_search_roots_puts_cwd_first_and_dedupes() {
        let cwd: SharedCwd = Arc::new(RwLock::new(PathBuf::from("/work/main")));
        let roots: SharedRoots = Arc::new(RwLock::new(vec![
            PathBuf::from("/work/shared"),
            PathBuf::from("/work/main"),
            PathBuf::from("/work/shared"),
            PathBuf::from("/work/docs"),
        ]));

        assert_eq!(search_roots(&cwd, &roots), vec![
            PathBuf::from("/work/main"),
            PathBuf::from("/work/shared"),
            PathBuf::from("/work/docs"),
        ]);
    }

    /// A root nested inside another is covered by it, so keeping both walks that tree twice and
    /// reports every file in it twice.
    #[test]
    fn test_search_roots_drops_roots_nested_in_another() {
        let cwd: SharedCwd = Arc::new(RwLock::new(PathBuf::from("/work/main")));
        let roots: SharedRoots = Arc::new(RwLock::new(vec![
            PathBuf::from("/work/main/nested"),
            PathBuf::from("/work/other"),
        ]));

        assert_eq!(search_roots(&cwd, &roots), vec![
            PathBuf::from("/work/main"),
            PathBuf::from("/work/other"),
        ]);
    }

    /// And the inverse: a root that *contains* `cwd` wins, because its walk already reaches
    /// everything under `cwd`. Dropping `cwd` from the search set is safe; it stays the base for
    /// relative paths and the shell either way.
    #[test]
    fn test_search_roots_lets_an_ancestor_root_subsume_cwd() {
        let cwd: SharedCwd = Arc::new(RwLock::new(PathBuf::from("/work/main")));
        let roots: SharedRoots = Arc::new(RwLock::new(vec![PathBuf::from("/work")]));
        assert_eq!(search_roots(&cwd, &roots), vec![PathBuf::from("/work")]);
    }

    /// A shared prefix is not containment: `/work/main2` is not inside `/work/main`.
    #[test]
    fn test_search_roots_keeps_sibling_with_shared_prefix() {
        let cwd: SharedCwd = Arc::new(RwLock::new(PathBuf::from("/work/main")));
        let roots: SharedRoots = Arc::new(RwLock::new(vec![PathBuf::from("/work/main2")]));
        assert_eq!(search_roots(&cwd, &roots), vec![
            PathBuf::from("/work/main"),
            PathBuf::from("/work/main2"),
        ]);
    }

    /// The single-root case has to stay exactly one path: that is every REPL and HTTP session, and
    /// every ACP client that sends no extra roots.
    #[test]
    fn test_search_roots_without_extras_is_just_cwd() {
        let cwd: SharedCwd = Arc::new(RwLock::new(PathBuf::from("/work/main")));
        let roots: SharedRoots = Arc::new(RwLock::new(Vec::new()));
        assert_eq!(search_roots(&cwd, &roots), vec![PathBuf::from(
            "/work/main"
        )]);
    }

    #[test]
    fn test_resolve_against_cwd_passes_absolute_paths_through() {
        let cwd: SharedCwd = Arc::new(RwLock::new(PathBuf::from("/home/agent")));
        let absolute = std::path::Path::new("/etc/hosts");
        let resolved = resolve_against_cwd(&cwd, absolute);
        assert_eq!(resolved, PathBuf::from("/etc/hosts"));
    }

    #[test]
    fn test_resolve_against_cwd_joins_relative_paths_to_session_cwd() {
        let cwd: SharedCwd = Arc::new(RwLock::new(PathBuf::from("/home/agent/project")));
        let resolved = resolve_against_cwd(&cwd, "src/main.rs");
        assert_eq!(resolved, PathBuf::from("/home/agent/project/src/main.rs"));
    }

    #[test]
    fn test_resolve_against_cwd_follows_subsequent_writes() {
        // Confirms multiple sessions in one process would observe their own cwds: a write to the
        // shared lock is visible on the next resolve, without touching process cwd.
        let cwd: SharedCwd = Arc::new(RwLock::new(PathBuf::from("/tmp/a")));
        let first = resolve_against_cwd(&cwd, "foo.txt");
        *cwd.write().expect("cwd lock") = PathBuf::from("/tmp/b");
        let second = resolve_against_cwd(&cwd, "foo.txt");
        assert_eq!(first, PathBuf::from("/tmp/a/foo.txt"));
        assert_eq!(second, PathBuf::from("/tmp/b/foo.txt"));
    }
}
