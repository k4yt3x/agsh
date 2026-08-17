//! The per-session workspace: where relative paths resolve from, and which roots a search sweeps.
//!
//! Its own module rather than a corner of `agent.rs` because every file-touching tool needs it and
//! nothing here needs an `Agent`. Living in `agent.rs` made `tools` depend on `agent` for a path
//! join, which is a cycle between the two largest modules in the tree and the reason a reader
//! looking for "how does `read_file` resolve a relative path" ended up in the turn loop.

use std::{
    path::PathBuf,
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

/// Workspace roots beyond [`SharedCwd`], as supplied by an ACP client's `additionalDirectories`.
///
/// A separate handle rather than a field on `SharedCwd` because only the two search tools and the
/// environment-context block care: widening [`resolve_against_cwd`] would touch every file tool's
/// constructor to serve two callers. `cwd` remains the base for relative paths, per the ACP spec,
/// so these expand *discovery* scope only.
///
/// Empty for the REPL, the HTTP API, and any ACP client that sends no extra roots.
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
    let mut kept: Vec<PathBuf> = Vec::new();
    for path in std::iter::once(cwd_snapshot(cwd)).chain(roots_snapshot(roots)) {
        if kept.iter().any(|existing| path.starts_with(existing)) {
            continue;
        }
        // This root is broader than ones already kept, so those become redundant.
        kept.retain(|existing| !existing.starts_with(&path));
        kept.push(path);
    }
    kept
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
