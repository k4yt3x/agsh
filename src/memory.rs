//! Agent memory: durable notes the agent writes for itself, surviving compaction and outliving any
//! one session.
//!
//! Each memory is a Markdown file at `~/.config/meka/memory/<name>.md` with YAML frontmatter
//! declaring a required `description` and an optional `priority`. The store is scoped to the meka
//! *instance* (i.e. to `MEKA_CONFIG_DIR`), not to a session or a directory: meka has no Project
//! concept, and the motivating deployment is a single always-on session reachable over chat, where
//! the agent is closer to a person than to a checkout.
//!
//! The shape deliberately mirrors [`crate::skills`] - discovery, an mtime-snapshot cache, and
//! frontmatter carrying a one-line description - because the retrieval problem is the same one:
//! advertise a cheap index in the per-turn context and load the body only on demand. What differs
//! is lifecycle. Skills are installed and then sit still; memories are written by the agent
//! constantly, so this module adds a write path and a priority used to rank the index when it
//! outgrows its budget.
//!
//! Why this survives compaction: the index rides [`crate::context::WorldSnapshot`], which
//! `Agent::last_rendered_world` re-states in full at session start, after every compaction, and
//! whenever the previous render scrolls out of the context window.

pub mod cli;

use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    sync::Arc,
    time::SystemTime,
};

use serde::Deserialize;
use tokio::sync::Mutex;

// Re-exported rather than referenced through `store` at each use site: priority is part of the
// memory store's public vocabulary (`meka memory add --priority`, the `memory_write` schema),
// and the constants moved to `store` only so `skills` could share the same scale.
pub use crate::store::{DEFAULT_PRIORITY, MAX_PRIORITY, MIN_PRIORITY, normalize_description};
use crate::store::{split_frontmatter, validate_entry_name, yaml_scalar};

/// A single durable note. `description` is what the agent sees every turn; `body` is fetched on
/// demand through the `memory_read` tool.
#[derive(Debug, Clone)]
pub struct Memory {
    pub name: String,
    pub description: String,
    pub priority: u8,
    pub path: PathBuf,
    /// Last-modified time, used to break priority ties (newest first) and to render a
    /// human-readable age in the index.
    pub mtime: SystemTime,
}

/// A file in the memory root that discovery could not turn into a [`Memory`].
///
/// Recorded rather than only logged, because the log is not a channel the model can read. From
/// inside a session a skipped file is indistinguishable from a memory nobody ever wrote: the index
/// does not list it and `memory_read` reports it missing. That silence is the failure worth
/// designing against, since it lets someone drop in a standing rule and believe it is in force for
/// as long as it takes them to look at stderr.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SkippedMemory {
    /// The file name as it appears on disk, e.g. `mica-policy.md`. A file name and not a memory
    /// name: an unusable name is itself one of the reasons a file lands here, and in that case
    /// there is no memory name to report.
    pub file: String,
    pub reason: String,
}

/// The outcome of one discovery pass: what parsed, and what did not.
#[derive(Debug, Clone, Default)]
pub struct MemoryIndex {
    /// Valid memories, in the order [`sort_for_index`] produced.
    pub memories: Vec<Memory>,
    /// Files that failed to parse, sorted by file name.
    ///
    /// Sorted because this reaches [`crate::context::WorldSnapshot`], which is compared by
    /// equality to decide whether the model needs telling anything. `read_dir` yields in
    /// filesystem order, so an unsorted list would re-announce the same skips at random.
    pub skipped: Vec<SkippedMemory>,
    /// How many valid memories the [`MAX_MEMORY_FILES`] cap dropped. Distinct from `skipped`:
    /// these parsed fine and were discarded for volume.
    pub ignored_over_cap: usize,
}

impl MemoryIndex {
    /// Why the file a memory of this name would live in was rejected, if it was.
    ///
    /// Every lookup that is about to report a name as absent checks this first. "No such memory"
    /// and "the file is right there and unreadable" call for opposite responses, and from the
    /// outside the two are indistinguishable: both are simply a name that is not in the index.
    pub fn skip_reason(&self, name: &str) -> Option<&str> {
        let file = memory_file_name(name);
        self.skipped
            .iter()
            .find(|skipped| skipped.file == file)
            .map(|skipped| skipped.reason.as_str())
    }
}

#[derive(Debug, Deserialize)]
struct Frontmatter {
    description: Option<String>,
    priority: Option<i64>,
}

/// Ceiling on how many files one discovery pass will parse. Bounds the per-turn cost of a memory
/// directory that has grown without anyone pruning it; the index budget in [`crate::context`]
/// trims further from there.
const MAX_MEMORY_FILES: usize = 500;

pub fn memory_dir() -> Option<PathBuf> {
    crate::config::meka_config_dir().map(|dir| dir.join("memory"))
}

/// The file name one memory lives under. The sole owner of the `<name>.md` layout convention, so
/// changing it is a one-line edit rather than a grep.
fn memory_file_name(name: &str) -> String {
    format!("{}.md", name)
}

/// Resolve one memory's path inside `root`. Performs no I/O and does not validate the name; callers
/// pair it with [`validate_memory_name`].
pub fn memory_file_in(root: &Path, name: &str) -> PathBuf {
    root.join(memory_file_name(name))
}

/// Validate that `name` is a safe filesystem-and-prompt-embeddable memory identifier. See
/// [`validate_entry_name`] for the rules and for why this is load-bearing rather than cosmetic:
/// the memory tools run at read permission, so the character class is what keeps `memory_write`
/// from being an arbitrary-file-write primitive.
pub fn validate_memory_name(name: &str) -> Result<(), String> {
    validate_entry_name(name, "memory")
}

/// Clamp a frontmatter `priority` for a memory. Thin wrapper over [`crate::store::parse_priority`]
/// that supplies this store's noun, mirroring [`validate_memory_name`].
pub fn parse_priority(raw: Option<i64>, name: &str) -> u8 {
    crate::store::parse_priority(raw, "memory", name)
}

/// Discover all valid memories in the user's memory directory. Returns an empty index if the
/// directory is missing or contains nothing valid.
pub fn discover_memories() -> MemoryIndex {
    let Some(root) = memory_dir() else {
        return MemoryIndex::default();
    };
    discover_memories_in(&root)
}

/// Walk a memory root and parse every `*.md`. Emits `tracing::warn!` for each malformed entry and
/// skips it, so one bad file never hides the rest of the store.
///
/// Memories come back in index order: `priority` ascending (lower is more important), then newest
/// first within a band, so a fresh note never outranks a standing rule merely for being recent.
///
/// Every skip is also *returned*, not just logged. See [`SkippedMemory`] for why the log alone was
/// not enough.
fn discover_memories_in(root: &Path) -> MemoryIndex {
    let entries = match std::fs::read_dir(root) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return MemoryIndex::default();
        }
        Err(error) => {
            tracing::warn!("failed to read memory dir {}: {}", root.display(), error);
            return MemoryIndex::default();
        }
    };

    let mut index = MemoryIndex::default();
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        // Dot-prefixed entries are editor swap files, `.DS_Store`, and the like - never memories.
        let Some(file_name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if file_name.starts_with('.') {
            continue;
        }
        let Some(name) = file_name.strip_suffix(".md") else {
            continue;
        };
        if let Err(reason) = validate_memory_name(name) {
            tracing::warn!("skipping memory file '{}': {}", file_name, reason);
            index.skipped.push(SkippedMemory {
                file: file_name.to_string(),
                reason,
            });
            continue;
        }

        match load_memory_definition(name, &path) {
            Ok(memory) => index.memories.push(memory),
            Err(reason) => {
                tracing::warn!("skipping memory '{}': {}", name, reason);
                index.skipped.push(SkippedMemory {
                    file: file_name.to_string(),
                    reason,
                });
            }
        }
    }

    // Sort *before* applying the cap. `read_dir` yields in filesystem order, so truncating first
    // would keep an arbitrary subset - a priority-0 standing rule could be dropped while a
    // priority-9 note survived, which defeats the point of ranking them.
    sort_for_index(&mut index.memories);
    if index.memories.len() > MAX_MEMORY_FILES {
        index.ignored_over_cap = index.memories.len() - MAX_MEMORY_FILES;
        tracing::warn!(
            "memory dir {} holds {} entries; keeping the {} highest-priority and ignoring {}",
            root.display(),
            index.memories.len(),
            MAX_MEMORY_FILES,
            index.ignored_over_cap
        );
        index.memories.truncate(MAX_MEMORY_FILES);
    }
    index
        .skipped
        .sort_by(|left, right| left.file.cmp(&right.file));
    index
}

/// Order memories the way the context index presents them: priority ascending, then newest first,
/// then by name so the result is total and stable across runs.
pub fn sort_for_index(memories: &mut [Memory]) {
    memories.sort_by(|a, b| {
        a.priority
            .cmp(&b.priority)
            .then_with(|| b.mtime.cmp(&a.mtime))
            .then_with(|| a.name.cmp(&b.name))
    });
}

fn load_memory_definition(name: &str, path: &Path) -> Result<Memory, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|error| format!("failed to read {}: {}", path.display(), error))?;
    let mtime = std::fs::metadata(path)
        .and_then(|metadata| metadata.modified())
        .unwrap_or(SystemTime::UNIX_EPOCH);
    parse_memory_definition(name, path, mtime, &content)
}

/// Parse a memory file's text into a [`Memory`]. Split out from [`load_memory_definition`] so the
/// write path can validate content in memory before it touches disk.
pub fn parse_memory_definition(
    name: &str,
    path: &Path,
    mtime: SystemTime,
    content: &str,
) -> Result<Memory, String> {
    let (frontmatter_str, _body) =
        split_frontmatter(content).ok_or_else(|| "missing YAML frontmatter".to_string())?;

    let frontmatter: Frontmatter = serde_norway::from_str(frontmatter_str)
        .map_err(|error| format!("invalid frontmatter: {}", error))?;

    let description = frontmatter
        .description
        .filter(|description| !description.trim().is_empty())
        .ok_or_else(|| "missing required field 'description'".to_string())?;

    Ok(Memory {
        name: name.to_string(),
        description: description.trim().to_string(),
        priority: parse_priority(frontmatter.priority, name),
        path: path.to_path_buf(),
        mtime,
    })
}

/// Read a memory's body (everything after the frontmatter). Unlike a skill body this gets no
/// variable substitution and no prepended header: a memory is recalled prose, not an instruction
/// sheet rooted in a bundle directory.
pub async fn load_memory_body(memory: &Memory) -> Result<String, String> {
    let content = tokio::fs::read_to_string(&memory.path)
        .await
        .map_err(|error| format!("failed to read {}: {}", memory.path.display(), error))?;

    Ok(split_frontmatter(&content)
        .map(|(_, body)| body.to_string())
        .unwrap_or(content))
}

/// Render a memory file: frontmatter followed by the body. `priority` is emitted only when it
/// differs from [`DEFAULT_PRIORITY`], so the common case stays a two-line header.
pub fn render_memory(description: &str, priority: u8, body: &str) -> String {
    use std::fmt::Write as _;

    let mut out = String::new();
    out.push_str("---\n");
    let _ = writeln!(
        out,
        "description: {}",
        yaml_scalar(&normalize_description(description))
    );
    if priority != DEFAULT_PRIORITY {
        let _ = writeln!(out, "priority: {}", priority);
    }
    out.push_str("---\n\n");
    out.push_str(body.trim_start_matches('\n'));
    if !out.ends_with('\n') {
        out.push('\n');
    }
    out
}

/// The body of an existing memory file, or `None` when there is nothing readable at `path`.
///
/// Falls back to the whole file when it carries no frontmatter, matching [`load_memory_body`]: such
/// a file is a body somebody wrote without a header, and giving it one is a repair. Discarding the
/// text instead would make a malformed memory worse the moment anyone tried to fix its metadata.
fn existing_body(path: &Path) -> Option<String> {
    let content = std::fs::read_to_string(path).ok()?;
    Some(
        split_frontmatter(&content)
            .map(|(_, body)| body.to_string())
            .unwrap_or(content),
    )
}

/// Create or overwrite a memory. Returns the path written.
///
/// `body` of `None` means "leave whatever is there alone", which is what a metadata-only update is:
/// a priority or description change should not cost the note its contents. Resolved here rather
/// than at each call site so no writer can get it wrong, because getting it wrong is silent.
/// `Some("")` is an explicit request to empty the body, and a `None` for a memory that does not
/// exist yet writes an empty one.
///
/// Validates the rendered file by parsing it back before it lands, so a description that would
/// break the frontmatter (an unescaped colon, say) fails loudly at write time rather than silently
/// dropping the memory out of every future index.
pub fn write_memory(
    root: &Path,
    name: &str,
    description: &str,
    priority: u8,
    body: Option<&str>,
) -> Result<PathBuf, String> {
    validate_memory_name(name)?;
    if description.trim().is_empty() {
        return Err("description cannot be empty".to_string());
    }

    let path = memory_file_in(root, name);
    // A validated name cannot escape the root, but a symlink already sitting at that name
    // redirects the write wherever it points. See [`crate::store::reject_symlinked_path`].
    crate::store::reject_symlinked_path(&path, "memory")?;
    let body = match body {
        Some(body) => body.to_string(),
        None => existing_body(&path).unwrap_or_default(),
    };
    let rendered = render_memory(description, priority, &body);
    parse_memory_definition(name, &path, SystemTime::now(), &rendered).map_err(|error| {
        format!("refusing to write a memory that would not parse back: {error}")
    })?;

    crate::config::write_file_atomic(&path, &rendered)
        .map_err(|error| format!("failed to write {}: {}", path.display(), error))?;
    Ok(path)
}

/// Human-readable age, e.g. "today", "yesterday", "47 days ago".
///
/// Deliberately not an ISO timestamp: models are poor at date arithmetic, and a rendered age
/// prompts staleness reasoning in a way a raw date does not. A memory is a point-in-time
/// observation, and the agent needs to weigh an old one accordingly.
pub fn render_age(mtime: SystemTime, now: SystemTime) -> String {
    let days = now
        .duration_since(mtime)
        .map(|elapsed| elapsed.as_secs() / 86_400)
        .unwrap_or(0);
    match days {
        0 => "today".to_string(),
        1 => "yesterday".to_string(),
        n => format!("{} days ago", n),
    }
}

/// Snapshot the disk state of a memory root: `path → mtime` for every non-dot `*.md`. Used by
/// [`MemoryCache`] to decide whether to re-run discovery.
///
/// Returns `None` when `read_dir` fails with anything other than `NotFound`; that signals the
/// caller to serve the cached (stale) state rather than wiping it on a transient filesystem
/// hiccup. `NotFound` maps to `Some(empty)` so a deleted memory dir properly clears the cache.
fn disk_snapshot(root: &Path) -> Option<BTreeMap<PathBuf, SystemTime>> {
    let entries = match std::fs::read_dir(root) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => {
            return Some(BTreeMap::new());
        }
        Err(_) => return None,
    };

    let mut map = BTreeMap::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_file() {
            continue;
        }
        let Some(file_name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if file_name.starts_with('.') || !file_name.ends_with(".md") {
            continue;
        }
        // Stat failure maps to UNIX_EPOCH so a later stat-success transition forces a diff.
        let mtime = std::fs::metadata(&path)
            .and_then(|metadata| metadata.modified())
            .unwrap_or(SystemTime::UNIX_EPOCH);
        map.insert(path, mtime);
    }
    Some(map)
}

/// Shared, atomically-swappable view of the memory list. Mirrors [`crate::skills::SkillCache`]:
/// reads perform a cheap mtime-snapshot check and only re-discover when the on-disk state actually
/// changed. The agent writes memories mid-session through the `memory_write` tool, so this is what
/// makes a just-written memory visible in the very next turn's index without a restart.
pub struct MemoryCache {
    /// Resolved memory root. `None` yields a permanently-empty cache, used when memory is disabled
    /// in config and by subcommands that don't read memories.
    root: Option<PathBuf>,
    /// Whether the subsystem is switched on at all, from `[memory] enabled`.
    ///
    /// Deliberately separate from `root`: a cache with no root is an *empty* store (nothing on
    /// disk, or test scaffolding), and its `memory_*` tools still belong in the registry. A
    /// disabled cache means the feature is off, so they are not registered and the `[Memory]`
    /// section never renders. Conflating the two made `meka tools list` hide tools that a real
    /// session would have had.
    enabled: bool,
    state: Mutex<CacheState>,
}

struct CacheState {
    index: Arc<MemoryIndex>,
    snapshot: BTreeMap<PathBuf, SystemTime>,
}

impl MemoryCache {
    /// Production constructor. Resolves [`memory_dir`] and seeds the cache.
    pub fn discover() -> Arc<Self> {
        Self::for_root(memory_dir())
    }

    /// Construct a cache backed by a specific root. `None` produces a permanently-empty cache.
    pub fn for_root(root: Option<PathBuf>) -> Arc<Self> {
        let (index, snapshot) = match root.as_deref() {
            Some(root) => (
                discover_memories_in(root),
                disk_snapshot(root).unwrap_or_default(),
            ),
            None => (MemoryIndex::default(), BTreeMap::new()),
        };
        Arc::new(Self {
            root,
            enabled: true,
            state: Mutex::new(CacheState {
                index: Arc::new(index),
                snapshot,
            }),
        })
    }

    /// A cache for a switched-off subsystem: empty, rootless, and reporting
    /// [`MemoryCache::enabled`] as `false` so the registration sites skip its tools.
    pub fn disabled() -> Arc<Self> {
        Arc::new(Self {
            root: None,
            enabled: false,
            state: Mutex::new(CacheState {
                index: Arc::new(MemoryIndex::default()),
                snapshot: BTreeMap::new(),
            }),
        })
    }

    /// Whether the subsystem is switched on. See the field docs on [`MemoryCache::enabled`].
    pub fn enabled(&self) -> bool {
        self.enabled
    }

    /// Return the current index, re-discovering first if the on-disk snapshot changed since the
    /// last call.
    ///
    /// One `Arc` covering both the memories and the skips, rather than a second accessor for the
    /// latter: two lookups would each re-check the snapshot and could straddle a rediscovery,
    /// leaving a caller reporting skips that no longer match the memories it listed.
    pub async fn current(&self) -> Arc<MemoryIndex> {
        let Some(root) = self.root.clone() else {
            return self.state.lock().await.index.clone();
        };
        // Discovery touches the filesystem and runs on every prompt from the async agent loop, so
        // offload it to the blocking pool. Transient errors yield `None`; serve stale state rather
        // than wipe the cache.
        let now = {
            let root = root.clone();
            match tokio::task::spawn_blocking(move || disk_snapshot(&root)).await {
                Ok(Some(snapshot)) => snapshot,
                _ => return self.state.lock().await.index.clone(),
            }
        };
        if self.state.lock().await.snapshot == now {
            return self.state.lock().await.index.clone();
        }
        // Run discovery *without* holding the state lock so concurrent callers aren't blocked
        // behind the filesystem walk. A racing caller may discover in parallel; harmless, since
        // both results derive from disk and the last write wins.
        let discovered =
            match tokio::task::spawn_blocking(move || discover_memories_in(&root)).await {
                Ok(index) => index,
                Err(error) => {
                    tracing::warn!("memory discovery task failed: {}", error);
                    return self.state.lock().await.index.clone();
                }
            };
        let mut state = self.state.lock().await;
        state.index = Arc::new(discovered);
        state.snapshot = now;
        state.index.clone()
    }

    /// The resolved root, for the tools and CLI that need to write into it.
    pub fn root(&self) -> Option<&Path> {
        self.root.as_deref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_memory_file(root: &Path, name: &str, content: &str) {
        std::fs::create_dir_all(root).expect("create memory dir");
        std::fs::write(memory_file_in(root, name), content).expect("write memory");
    }

    fn frontmatter(description: &str) -> String {
        format!("---\ndescription: {}\n---\nBody\n", description)
    }

    /// Bump the mtime far enough into the future to defeat 1-second filesystem resolution.
    fn bump_mtime(path: &Path) {
        let file = std::fs::OpenOptions::new()
            .write(true)
            .open(path)
            .expect("open for mtime bump");
        let future = SystemTime::now() + std::time::Duration::from_secs(10);
        file.set_modified(future).expect("set_modified");
    }

    #[test]
    fn test_load_valid_memory() {
        let temp = tempfile::tempdir().expect("tempdir");
        write_memory_file(temp.path(), "a-note", &frontmatter("A durable fact"));

        let index = discover_memories_in(temp.path());
        assert_eq!(index.memories.len(), 1);
        assert_eq!(index.memories[0].name, "a-note");
        assert_eq!(index.memories[0].description, "A durable fact");
        assert_eq!(index.memories[0].priority, DEFAULT_PRIORITY);
        assert!(index.skipped.is_empty());
    }

    #[test]
    fn test_missing_description_is_skipped() {
        let temp = tempfile::tempdir().expect("tempdir");
        write_memory_file(temp.path(), "bad", "---\npriority: 1\n---\nBody\n");
        write_memory_file(temp.path(), "good", &frontmatter("fine"));

        // One malformed file must not hide the rest of the store.
        let index = discover_memories_in(temp.path());
        assert_eq!(index.memories.len(), 1);
        assert_eq!(index.memories[0].name, "good");
        assert_eq!(index.skipped.len(), 1);
        assert_eq!(index.skipped[0].file, "bad.md");
    }

    /// The skip list is the whole point of recording rather than only logging: a file nobody can
    /// read has to be nameable from inside a session, with the reason attached, or it is
    /// indistinguishable from a memory that was never written.
    #[test]
    fn test_every_unreadable_file_is_named_with_its_reason() {
        let temp = tempfile::tempdir().expect("tempdir");
        write_memory_file(temp.path(), "no-frontmatter", "just prose, no header\n");
        write_memory_file(
            temp.path(),
            "no-description",
            "---\npriority: 1\n---\nBody\n",
        );
        write_memory_file(temp.path(), "bad-yaml", "---\ndescription: [\n---\n");
        write_memory_file(temp.path(), "has space", &frontmatter("unusable name"));
        write_memory_file(temp.path(), "fine", &frontmatter("ok"));

        let index = discover_memories_in(temp.path());
        assert_eq!(index.memories.len(), 1);

        // Sorted by file name, so the same store always renders the same section: `read_dir`
        // order would otherwise re-announce these at random.
        let files: Vec<&str> = index
            .skipped
            .iter()
            .map(|skipped| skipped.file.as_str())
            .collect();
        assert_eq!(files, vec![
            "bad-yaml.md",
            "has space.md",
            "no-description.md",
            "no-frontmatter.md"
        ]);

        assert_eq!(
            index.skip_reason("no-frontmatter"),
            Some("missing YAML frontmatter")
        );
        assert_eq!(
            index.skip_reason("no-description"),
            Some("missing required field 'description'")
        );
        assert!(
            index
                .skip_reason("bad-yaml")
                .is_some_and(|reason| reason.starts_with("invalid frontmatter")),
            "{:?}",
            index.skip_reason("bad-yaml")
        );
        // Disqualified by its name, so the reason has to come from name validation rather than
        // from a parse that never ran.
        assert!(
            index
                .skip_reason("has space")
                .is_some_and(|reason| reason.contains("invalid character")),
            "{:?}",
            index.skip_reason("has space")
        );
        assert_eq!(index.skip_reason("fine"), None);
        assert_eq!(index.skip_reason("never-existed"), None);
    }

    #[test]
    fn test_non_markdown_and_dotfiles_ignored() {
        let temp = tempfile::tempdir().expect("tempdir");
        write_memory_file(temp.path(), "real", &frontmatter("x"));
        std::fs::write(temp.path().join("notes.txt"), "plain").expect("write txt");
        std::fs::write(temp.path().join(".hidden.md"), frontmatter("y")).expect("write dotfile");

        let index = discover_memories_in(temp.path());
        assert_eq!(index.memories.len(), 1);
        assert_eq!(index.memories[0].name, "real");
        // Neither is a memory that failed, so neither belongs in the skip list: reporting a
        // `.DS_Store` as an unreadable memory would train the reader to ignore the section.
        assert!(index.skipped.is_empty());
    }

    /// Without this check `memory_write` would be an arbitrary-file-write primitive, because the
    /// memory tools are reachable at `Permission::Read`.
    #[test]
    fn test_validate_memory_name_rejects_traversal_and_separators() {
        for bad in [
            "",
            "..",
            "../escape",
            "a/b",
            "a\\b",
            "/abs",
            ".hidden",
            "-leading",
            "has space",
            "has:colon",
        ] {
            assert!(
                validate_memory_name(bad).is_err(),
                "'{bad}' must be rejected"
            );
        }
        for good in ["a", "note", "a-note_2", "K4YT3X-prefers-terse"] {
            assert!(
                validate_memory_name(good).is_ok(),
                "'{good}' must be accepted"
            );
        }
        assert!(validate_memory_name(&"a".repeat(crate::store::MAX_ENTRY_NAME_LEN + 1)).is_err());
    }

    #[test]
    fn test_priority_defaults_and_clamps() {
        assert_eq!(parse_priority(None, "x"), DEFAULT_PRIORITY);
        assert_eq!(parse_priority(Some(0), "x"), 0);
        assert_eq!(parse_priority(Some(9), "x"), 9);
        // Out of range is clamped, not rejected: a nonsense priority must not make the memory
        // itself unreachable.
        assert_eq!(parse_priority(Some(-3), "x"), MIN_PRIORITY);
        assert_eq!(parse_priority(Some(1000), "x"), MAX_PRIORITY);
    }

    /// Priority first, then newest within a band. A fresh low-priority note must not displace a
    /// standing rule at the top of the index.
    #[test]
    fn test_index_order_is_priority_then_recency() {
        let temp = tempfile::tempdir().expect("tempdir");
        write_memory_file(
            temp.path(),
            "standing",
            "---\ndescription: rule\npriority: 1\n---\n",
        );
        write_memory_file(temp.path(), "old-default", &frontmatter("older"));
        write_memory_file(temp.path(), "new-default", &frontmatter("newer"));
        bump_mtime(&temp.path().join("new-default.md"));

        let names: Vec<String> = discover_memories_in(temp.path())
            .memories
            .into_iter()
            .map(|memory| memory.name)
            .collect();
        assert_eq!(names, vec!["standing", "new-default", "old-default"]);
    }

    /// The cap has to drop the *least important* memories, not whichever ones `read_dir` happened
    /// to yield last. Truncating before the sort meant a priority-0 standing rule could vanish
    /// while priority-9 noise survived, which defeats the point of ranking them at all.
    #[test]
    fn test_discovery_cap_keeps_the_highest_priority() {
        let temp = tempfile::tempdir().expect("tempdir");
        // One standing rule among MAX_MEMORY_FILES + 50 low-priority notes. Named `zzz-` so
        // filesystem order is very unlikely to favour it.
        write_memory_file(
            temp.path(),
            "zzz-standing-rule",
            "---\ndescription: always applies\npriority: 0\n---\n",
        );
        for index in 0..(MAX_MEMORY_FILES + 50) {
            write_memory_file(
                temp.path(),
                &format!("filler-{index:04}"),
                "---\ndescription: noise\npriority: 9\n---\n",
            );
        }

        let index = discover_memories_in(temp.path());
        assert_eq!(index.memories.len(), MAX_MEMORY_FILES);
        assert_eq!(
            index.memories[0].name, "zzz-standing-rule",
            "the highest-priority memory must survive the cap and lead the index"
        );
        // Counted, not just logged: a store trimmed by the cap looks exactly like a store that
        // small from inside a session.
        assert_eq!(index.ignored_over_cap, 51);
    }

    /// A description is a single line by contract. Left to YAML, a newline would fold to a space
    /// on the next read and the stored value would quietly differ from what was requested.
    #[test]
    fn test_description_is_normalised_to_one_line() {
        assert_eq!(
            normalize_description("line one\nline two"),
            "line one line two"
        );
        assert_eq!(normalize_description("  padded   out  "), "padded out");

        // A blank line is the case that bites: YAML folds "a\n\nb" back to a *literal* newline,
        // which would then be rendered mid-way through an index entry and break the one-line-per
        // -memory shape the `[Memory]` section and its budget both assume.
        let temp = tempfile::tempdir().expect("tempdir");
        write_memory(temp.path(), "n", "para one\n\npara two", 5, Some("body")).expect("write");
        let index = discover_memories_in(temp.path());
        assert_eq!(index.memories[0].description, "para one para two");
        assert!(
            !index.memories[0].description.contains('\n'),
            "a description must never carry a newline into the index"
        );
    }

    #[test]
    fn test_render_memory_round_trips() {
        let rendered = render_memory("note: with a colon", 2, "The body\n");
        let parsed = parse_memory_definition(
            "x",
            Path::new("/tmp/x.md"),
            SystemTime::UNIX_EPOCH,
            &rendered,
        )
        .expect("round trip");
        assert_eq!(parsed.description, "note: with a colon");
        assert_eq!(parsed.priority, 2);
    }

    /// The default priority is omitted from the file so the common case stays a two-line header.
    #[test]
    fn test_render_memory_omits_default_priority() {
        let rendered = render_memory("plain", DEFAULT_PRIORITY, "body");
        assert!(!rendered.contains("priority:"), "{rendered}");
        assert!(render_memory("plain", 1, "body").contains("priority: 1"));
    }

    #[test]
    fn test_write_memory_rejects_bad_name_and_empty_description() {
        let temp = tempfile::tempdir().expect("tempdir");
        assert!(write_memory(temp.path(), "../escape", "d", 5, Some("b")).is_err());
        assert!(write_memory(temp.path(), "ok", "   ", 5, Some("b")).is_err());
        assert!(write_memory(temp.path(), "ok", "d", 5, Some("b")).is_ok());
    }

    /// A metadata-only write must not cost the note its contents. `body` has always been optional,
    /// so this is the call the API invites, and rendering the absence as an empty body made a
    /// priority change silently delete everything the memory said.
    /// `validate_memory_name` stops a name from escaping the root, but it cannot see a symlink
    /// already sitting at that name. Following one writes outside the store at *read* permission,
    /// whose whole contract is that nothing in the user's tree changes. Archives preserve symlinks,
    /// so unpacking something into the memory directory is enough to plant one.
    #[cfg(unix)]
    #[test]
    fn test_write_memory_refuses_a_symlinked_path() {
        let temp = tempfile::tempdir().expect("tempdir");
        let root = temp.path().join("memory");
        let victim = temp.path().join("victim.txt");
        std::fs::create_dir_all(&root).expect("root");
        std::fs::write(&victim, "ORIGINAL").expect("victim");
        std::os::unix::fs::symlink(&victim, root.join("evil.md")).expect("symlink");

        let error = write_memory(&root, "evil", "d", 5, Some("PWNED"))
            .expect_err("must refuse to write through a symlink");
        assert!(error.contains("symlink"), "{error}");
        assert_eq!(
            std::fs::read_to_string(&victim).expect("read"),
            "ORIGINAL",
            "the target must be untouched"
        );
    }

    #[test]
    fn test_write_memory_keeps_an_omitted_body() {
        let temp = tempfile::tempdir().expect("tempdir");
        write_memory(
            temp.path(),
            "note",
            "first",
            5,
            Some("Detail worth keeping.\n"),
        )
        .expect("initial write");

        write_memory(temp.path(), "note", "revised", 1, None).expect("metadata-only write");
        let index = discover_memories_in(temp.path());
        assert_eq!(index.memories[0].description, "revised");
        assert_eq!(index.memories[0].priority, 1);
        let content =
            std::fs::read_to_string(memory_file_in(temp.path(), "note")).expect("read back");
        assert!(content.contains("Detail worth keeping."), "{content}");

        // Clearing stays possible, just no longer the accident.
        write_memory(temp.path(), "note", "revised", 1, Some("")).expect("explicit clear");
        let content =
            std::fs::read_to_string(memory_file_in(temp.path(), "note")).expect("read back");
        assert!(!content.contains("Detail worth keeping."), "{content}");
    }

    /// An omitted body on a name with no file yet is a create, not a failure.
    #[test]
    fn test_write_memory_with_no_body_creates_an_empty_one() {
        let temp = tempfile::tempdir().expect("tempdir");
        write_memory(temp.path(), "fresh", "d", 5, None).expect("write");
        let index = discover_memories_in(temp.path());
        assert_eq!(index.memories.len(), 1);
        assert_eq!(index.memories[0].name, "fresh");
    }

    /// A file with no frontmatter is a body somebody wrote without a header. Giving it one repairs
    /// it; discarding the text would make a broken memory worse the moment anyone tried to fix it.
    #[test]
    fn test_write_memory_adopts_a_frontmatterless_file_as_its_body() {
        let temp = tempfile::tempdir().expect("tempdir");
        write_memory_file(temp.path(), "orphan", "Prose that never had a header.\n");

        write_memory(temp.path(), "orphan", "now described", 5, None).expect("write");
        let index = discover_memories_in(temp.path());
        assert_eq!(index.memories.len(), 1);
        assert!(index.skipped.is_empty(), "the repair must clear the skip");
        let body = std::fs::read_to_string(memory_file_in(temp.path(), "orphan")).expect("read");
        assert!(body.contains("Prose that never had a header."), "{body}");
    }

    #[test]
    fn test_render_age() {
        let now = SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(100 * 86_400);
        assert_eq!(render_age(now, now), "today");
        assert_eq!(
            render_age(now - std::time::Duration::from_secs(86_400), now),
            "yesterday"
        );
        assert_eq!(
            render_age(now - std::time::Duration::from_secs(47 * 86_400), now),
            "47 days ago"
        );
        // A future mtime (clock skew) must not panic or underflow.
        assert_eq!(
            render_age(now + std::time::Duration::from_secs(86_400), now),
            "today"
        );
    }

    #[tokio::test]
    async fn test_memory_cache_picks_up_new_memory() {
        let temp = tempfile::tempdir().expect("tempdir");
        let cache = MemoryCache::for_root(Some(temp.path().to_path_buf()));
        assert!(cache.current().await.memories.is_empty());

        write_memory_file(temp.path(), "foo", &frontmatter("first"));
        let index = cache.current().await;
        assert_eq!(index.memories.len(), 1);
        assert_eq!(index.memories[0].name, "foo");
    }

    #[tokio::test]
    async fn test_memory_cache_detects_modification() {
        let temp = tempfile::tempdir().expect("tempdir");
        write_memory_file(temp.path(), "foo", &frontmatter("old"));
        let cache = MemoryCache::for_root(Some(temp.path().to_path_buf()));
        assert_eq!(cache.current().await.memories[0].description, "old");

        let path = temp.path().join("foo.md");
        std::fs::write(&path, frontmatter("new")).expect("rewrite");
        bump_mtime(&path);

        assert_eq!(cache.current().await.memories[0].description, "new");
    }

    /// A file that breaks after the cache was seeded has to reach the next reader. Discovery only
    /// re-runs when the disk snapshot moves, so a skip recorded once and never revisited would go
    /// unmentioned for the rest of the session.
    #[tokio::test]
    async fn test_memory_cache_picks_up_a_newly_broken_file() {
        let temp = tempfile::tempdir().expect("tempdir");
        write_memory_file(temp.path(), "foo", &frontmatter("fine"));
        let cache = MemoryCache::for_root(Some(temp.path().to_path_buf()));
        assert!(cache.current().await.skipped.is_empty());

        let path = temp.path().join("foo.md");
        std::fs::write(&path, "someone deleted the frontmatter\n").expect("rewrite");
        bump_mtime(&path);

        let index = cache.current().await;
        assert!(index.memories.is_empty());
        assert_eq!(index.skip_reason("foo"), Some("missing YAML frontmatter"));
    }

    #[tokio::test]
    async fn test_memory_cache_drops_removed_memory() {
        let temp = tempfile::tempdir().expect("tempdir");
        write_memory_file(temp.path(), "foo", &frontmatter("first"));
        let cache = MemoryCache::for_root(Some(temp.path().to_path_buf()));
        assert_eq!(cache.current().await.memories.len(), 1);

        std::fs::remove_file(temp.path().join("foo.md")).expect("rm");
        assert!(cache.current().await.memories.is_empty());
    }

    #[tokio::test]
    async fn test_memory_cache_stable_when_unchanged() {
        let temp = tempfile::tempdir().expect("tempdir");
        write_memory_file(temp.path(), "foo", &frontmatter("first"));
        let cache = MemoryCache::for_root(Some(temp.path().to_path_buf()));

        let first = cache.current().await;
        let second = cache.current().await;
        // Same Arc ⇒ no rediscovery, proving the stable-snapshot path really skips the walk.
        assert!(Arc::ptr_eq(&first, &second));
    }

    #[tokio::test]
    async fn test_memory_cache_with_no_root_is_empty() {
        let cache = MemoryCache::for_root(None);
        assert!(cache.current().await.memories.is_empty());
        assert!(cache.root().is_none());
    }

    #[tokio::test]
    async fn test_load_memory_body_strips_frontmatter() {
        let temp = tempfile::tempdir().expect("tempdir");
        write_memory_file(
            temp.path(),
            "foo",
            "---\ndescription: d\n---\nLine one\nLine two\n",
        );
        let index = discover_memories_in(temp.path());
        let body = load_memory_body(&index.memories[0]).await.expect("body");
        assert_eq!(body, "Line one\nLine two\n");
    }
}
