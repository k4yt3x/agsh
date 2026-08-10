//! Skill discovery and loading. Walks `~/.config/meka/skills/<name>/SKILL.md`, parses the YAML
//! frontmatter (`description`, `version`, `author`, `source_url`; unknown keys are ignored), and
//! exposes the resulting [`Skill`] structs to the agent for system-prompt injection and `skill`
//! tool dispatch.

pub mod cli;

use std::{
    collections::BTreeMap,
    path::{Path, PathBuf},
    sync::Arc,
    time::SystemTime,
};

use serde::Deserialize;
use tokio::sync::Mutex;

use crate::store::{split_frontmatter, validate_entry_name, yaml_scalar};

#[derive(Debug, Clone)]
pub struct Skill {
    pub name: String,
    pub source_dir: PathBuf,
    pub description: String,
    pub version: Option<String>,
    /// Optional attribution string. Informational only.
    pub author: Option<String>,
    /// Optional `https://` URL the skill's `SKILL.md` can be re-fetched from. When set, `meka skill
    /// update` can refresh the skill in place. `None` skills are skipped by `update`.
    pub source_url: Option<String>,
    pub body_path: PathBuf,
}

#[derive(Debug, Deserialize)]
struct Frontmatter {
    description: Option<String>,
    version: Option<String>,
    author: Option<String>,
    source_url: Option<String>,
}

pub fn skills_dir() -> Option<PathBuf> {
    crate::config::meka_config_dir().map(|dir| dir.join("skills"))
}

/// Discover all valid skills in the user's skills directory. Returns an empty vec if the directory
/// is missing or contains no valid skills.
pub fn discover_skills() -> Vec<Skill> {
    let Some(root) = skills_dir() else {
        return Vec::new();
    };
    discover_skills_in(&root)
}

/// Walk a specific skills root and parse every `SKILL.md`. Emits `tracing::warn!` for each
/// malformed entry; that warning behavior is the signal the [`SkillCache`] relies on to surface
/// broken-skill notices at startup and only re-fire when the on-disk snapshot changes.
fn discover_skills_in(root: &Path) -> Vec<Skill> {
    let entries = match std::fs::read_dir(root) {
        Ok(entries) => entries,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => return Vec::new(),
        Err(error) => {
            tracing::warn!("failed to read skills dir {}: {}", root.display(), error);
            return Vec::new();
        }
    };

    let mut skills = Vec::new();
    for entry in entries.flatten() {
        let path = entry.path();
        if !path.is_dir() {
            continue;
        }

        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        // Skip any dot-prefixed entry: VCS metadata (`.git`), editor/IDE state (`.vscode`,
        // `.idea`), filesystem artifacts (`.DS_Store`), etc. None are real skills, and silently
        // skipping them avoids spurious "missing SKILL.md" warnings.
        if name.starts_with('.') {
            continue;
        }
        let name = name.to_string();

        let skill_file = path.join("SKILL.md");
        match load_skill_definition(&name, &path, &skill_file) {
            Ok(skill) => skills.push(skill),
            Err(error) => {
                tracing::warn!("skipping skill '{}': {}", name, error);
            }
        }
    }

    skills.sort_by(|a, b| a.name.cmp(&b.name));
    skills
}

/// Snapshot the disk state of a skills root: `subdir/SKILL.md → mtime` for every non-dot
/// subdirectory. Used by [`SkillCache`] to decide whether to re-run discovery on the next turn.
///
/// Returns `None` when `read_dir` fails with anything other than `NotFound`; that signals the
/// caller to serve the cached (stale) state rather than wiping it on a transient filesystem hiccup.
/// A `NotFound` error maps to `Some(empty)` so a deleted skills dir properly clears the cache.
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
        if !path.is_dir() {
            continue;
        }
        let Some(name) = path.file_name().and_then(|name| name.to_str()) else {
            continue;
        };
        if name.starts_with('.') {
            continue;
        }
        let skill_file = path.join("SKILL.md");
        // Stat failure (file missing, perm denied) maps to UNIX_EPOCH so a later stat-success
        // transition forces a snapshot diff and reload.
        let mtime = std::fs::metadata(&skill_file)
            .and_then(|metadata| metadata.modified())
            .unwrap_or(SystemTime::UNIX_EPOCH);
        map.insert(skill_file, mtime);
    }
    Some(map)
}

/// Shared, atomically-swappable view of the skill list. Construction runs an initial
/// [`discover_skills_in`] pass so broken-skill warnings surface during agent startup (above the
/// first REPL prompt) instead of during the first turn. Subsequent reads via
/// [`SkillCache::current`] perform a cheap mtime-snapshot check and only re-discover when the
/// on-disk state actually changed; identical broken-skill warnings naturally dedup across turns
/// because the inner walk is skipped when the snapshot is stable.
pub struct SkillCache {
    /// Resolved skills root. `None` when [`skills_dir`] returns `None` or when constructed via
    /// `SkillCache::for_root(None)` for test scaffolding / subcommands that don't read skills.
    root: Option<PathBuf>,
    /// Whether the subsystem is switched on at all, from `[skills] enabled`.
    ///
    /// Deliberately separate from `root`: a cache with no root is an *empty* store (nothing on
    /// disk, or test scaffolding), and its `skill` tool still belong in the registry. A disabled
    /// cache means the feature is off, so they are not registered and the `[Skills]` section
    /// never renders. Conflating the two made `meka tools list` hide tools that a real session
    /// would have had.
    enabled: bool,
    state: Mutex<CacheState>,
}

struct CacheState {
    skills: Arc<Vec<Skill>>,
    snapshot: BTreeMap<PathBuf, SystemTime>,
}

impl SkillCache {
    /// Production constructor. Resolves [`skills_dir`] and seeds the cache.
    pub fn discover() -> Arc<Self> {
        Self::for_root(skills_dir())
    }

    /// Construct a cache backed by a specific root. `None` produces a permanently-empty cache,
    /// useful for tests and for subcommands (`meka tools list`) that don't read skill metadata.
    pub fn for_root(root: Option<PathBuf>) -> Arc<Self> {
        let (skills, snapshot) = match root.as_deref() {
            Some(root) => (
                discover_skills_in(root),
                disk_snapshot(root).unwrap_or_default(),
            ),
            None => (Vec::new(), BTreeMap::new()),
        };
        Arc::new(Self {
            root,
            enabled: true,
            state: Mutex::new(CacheState {
                skills: Arc::new(skills),
                snapshot,
            }),
        })
    }

    /// A cache for a switched-off subsystem: empty, rootless, and reporting
    /// [`SkillCache::enabled`] as `false` so the registration sites skip its tools.
    pub fn disabled() -> Arc<Self> {
        Arc::new(Self {
            root: None,
            enabled: false,
            state: Mutex::new(CacheState {
                skills: Arc::new(Vec::new()),
                snapshot: BTreeMap::new(),
            }),
        })
    }

    /// Whether the subsystem is switched on. See the field docs on [`SkillCache::enabled`].
    pub fn enabled(&self) -> bool {
        self.enabled
    }

    /// Return the current skill list, re-discovering first if the on-disk snapshot has changed
    /// since the last call. Cheap when nothing changed: one `read_dir` + N `metadata()` calls and a
    /// `BTreeMap` comparison, then an `Arc::clone` of the cached vec.
    pub async fn current(&self) -> Arc<Vec<Skill>> {
        let Some(root) = self.root.clone() else {
            return self.state.lock().await.skills.clone();
        };
        // Discovery touches the filesystem (`read_dir` + per-skill `metadata` / `read_to_string`);
        // this runs on every prompt from the async agent loop, so offload it to the blocking pool.
        // Transient errors (e.g. EACCES on the dir) yield `None`; serve stale state rather than
        // wipe the cache.
        let now = {
            let root = root.clone();
            match tokio::task::spawn_blocking(move || disk_snapshot(&root)).await {
                Ok(Some(snapshot)) => snapshot,
                _ => return self.state.lock().await.skills.clone(),
            }
        };
        if self.state.lock().await.snapshot == now {
            return self.state.lock().await.skills.clone();
        }
        // Run discovery *without* holding the state lock so concurrent `current()` callers aren't
        // blocked behind the filesystem walk. A racing caller may discover in parallel. Harmless:
        // both results derive from disk and the last write wins.
        let discovered = match tokio::task::spawn_blocking(move || discover_skills_in(&root)).await
        {
            Ok(skills) => skills,
            Err(error) => {
                tracing::warn!("skill discovery task failed: {}", error);
                return self.state.lock().await.skills.clone();
            }
        };
        let mut state = self.state.lock().await;
        state.skills = Arc::new(discovered);
        state.snapshot = now;
        state.skills.clone()
    }
}

fn load_skill_definition(
    name: &str,
    source_dir: &Path,
    skill_file: &Path,
) -> Result<Skill, String> {
    let content = std::fs::read_to_string(skill_file)
        .map_err(|error| format!("failed to read {}: {}", skill_file.display(), error))?;
    parse_skill_definition(name, source_dir, skill_file, &content)
}

/// Parse a `SKILL.md`'s text into a [`Skill`]. Split out from [`load_skill_definition`] so `meka
/// skill update` can validate fetched content in memory before it touches the on-disk file.
pub fn parse_skill_definition(
    name: &str,
    source_dir: &Path,
    skill_file: &Path,
    content: &str,
) -> Result<Skill, String> {
    let (frontmatter_str, _body) =
        split_frontmatter(content).ok_or_else(|| "missing YAML frontmatter".to_string())?;

    let frontmatter: Frontmatter = serde_norway::from_str(frontmatter_str)
        .map_err(|error| format!("invalid frontmatter: {}", error))?;

    let description = frontmatter
        .description
        .filter(|description| !description.trim().is_empty())
        .ok_or_else(|| "missing required field 'description'".to_string())?;

    Ok(Skill {
        name: name.to_string(),
        source_dir: source_dir.to_path_buf(),
        description,
        version: frontmatter.version,
        author: frontmatter.author,
        source_url: frontmatter.source_url,
        body_path: skill_file.to_path_buf(),
    })
}

/// Load the body (post-frontmatter) of a skill, perform variable substitution, and prepend the
/// [`skill_context_header`] so every consumer (the `skill` tool, `--skill`, `/skill`,
/// `spawn_agent`'s skill delegation, and `meka skill show`) sees the skill's base directory.
pub async fn load_skill_body(skill: &Skill, session_id: Option<&str>) -> Result<String, String> {
    let content = tokio::fs::read_to_string(&skill.body_path)
        .await
        .map_err(|error| format!("failed to read {}: {}", skill.body_path.display(), error))?;

    let body = split_frontmatter(&content)
        .map(|(_, body)| body.to_string())
        .unwrap_or(content);

    Ok(format!(
        "{}\n\n{}",
        skill_context_header(skill),
        substitute_variables(&body, skill, session_id)
    ))
}

/// Build the one-line context header prepended to a skill body by [`load_skill_body`]. Points the
/// agent at the skill's directory so bare filenames in the body (bundled scripts, data files)
/// resolve correctly without the author having to spell out `${MEKA_SKILL_DIR}` on every reference.
fn skill_context_header(skill: &Skill) -> String {
    format!(
        "Base directory for this skill and its bundled files: {}",
        skill.source_dir.display()
    )
}

fn substitute_variables(text: &str, skill: &Skill, session_id: Option<&str>) -> String {
    let mut result = text.replace("${MEKA_SKILL_DIR}", &skill.source_dir.display().to_string());
    if let Some(id) = session_id {
        result = result.replace("${MEKA_SESSION_ID}", id);
    }
    result
}

/// Validate that `name` is a safe filesystem-and-prompt-embeddable skill identifier. See
/// [`validate_entry_name`] for the rules; rejecting the character class outright is what keeps a
/// name from escaping the skills directory or breaking the slash-command grammar.
pub fn validate_skill_name(name: &str) -> Result<(), String> {
    validate_entry_name(name, "skill")
}

/// Resolve `~/.config/meka/skills/<name>` for a given skill name. Returns `None` if the meka config
/// directory cannot be determined. Performs no I/O and does not validate the name; callers are
/// expected to call [`validate_skill_name`] first.
pub fn skill_dir_for(name: &str) -> Option<PathBuf> {
    skills_dir().map(|root| root.join(name))
}

/// Render the default `SKILL.md` template for a new skill. Optional fields are emitted only when
/// set, so the resulting file stays as minimal as the user's input.
pub fn render_template(
    name: &str,
    description: &str,
    version: Option<&str>,
    author: Option<&str>,
    source_url: Option<&str>,
) -> String {
    use std::fmt::Write as _;

    let mut out = String::new();
    out.push_str("---\n");
    let _ = writeln!(out, "description: {}", yaml_scalar(description));
    if let Some(v) = version {
        let _ = writeln!(out, "version: {}", yaml_scalar(v));
    }
    if let Some(a) = author {
        let _ = writeln!(out, "author: {}", yaml_scalar(a));
    }
    if let Some(url) = source_url {
        let _ = writeln!(out, "source_url: {}", yaml_scalar(url));
    }
    out.push_str("---\n\n");
    let _ = writeln!(out, "# {}", name);
    out.push('\n');
    out.push_str(
        "Skill body. Use `${MEKA_SKILL_DIR}` to reference files bundled in this skill's\n\
         directory, or `${MEKA_SESSION_ID}` for the active session UUID.\n",
    );
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    fn write_skill(root: &Path, name: &str, skill_md: &str) {
        let dir = root.join(name);
        std::fs::create_dir_all(&dir).expect("create skill dir");
        std::fs::write(dir.join("SKILL.md"), skill_md).expect("write SKILL.md");
    }

    #[test]
    fn test_load_valid_skill() {
        let temp = tempfile::tempdir().expect("tempdir");
        write_skill(
            temp.path(),
            "test-skill",
            "---\ndescription: A test skill\n---\nBody content\n",
        );

        let skill_path = temp.path().join("test-skill");
        let skill_file = skill_path.join("SKILL.md");
        let skill =
            load_skill_definition("test-skill", &skill_path, &skill_file).expect("should load");

        assert_eq!(skill.name, "test-skill");
        assert_eq!(skill.description, "A test skill");
        assert!(skill.version.is_none());
        assert!(skill.author.is_none());
        assert!(skill.source_url.is_none());
    }

    #[test]
    fn test_load_skill_with_all_fields() {
        let temp = tempfile::tempdir().expect("tempdir");
        write_skill(
            temp.path(),
            "full-skill",
            "---\n\
             description: Complete skill\n\
             version: \"1.2\"\n\
             author: John Doe <john.doe@example.com>\n\
             source_url: https://example.com/SKILL.md\n\
             ---\nBody\n",
        );

        let skill_path = temp.path().join("full-skill");
        let skill = load_skill_definition("full-skill", &skill_path, &skill_path.join("SKILL.md"))
            .expect("should load");

        assert_eq!(skill.version.as_deref(), Some("1.2"));
        assert_eq!(
            skill.author.as_deref(),
            Some("John Doe <john.doe@example.com>")
        );
        assert_eq!(
            skill.source_url.as_deref(),
            Some("https://example.com/SKILL.md")
        );
    }

    #[test]
    fn test_unknown_frontmatter_keys_are_ignored() {
        // Skills authored for Claude Code carry keys meka doesn't model (when_to_use,
        // allowed-tools, hooks, ...). serde ignores unknown fields, so such a skill still parses on
        // a `description`.
        let temp = tempfile::tempdir().expect("tempdir");
        write_skill(
            temp.path(),
            "cc-skill",
            "---\n\
             description: A CC-shaped skill\n\
             when_to_use: legacy field\n\
             allowed-tools: [read_file]\n\
             user-invocable: false\n\
             ---\nBody\n",
        );

        let skill_path = temp.path().join("cc-skill");
        let skill = load_skill_definition("cc-skill", &skill_path, &skill_path.join("SKILL.md"))
            .expect("unknown keys must not break parsing");
        assert_eq!(skill.description, "A CC-shaped skill");
    }

    #[test]
    fn test_missing_description_rejected() {
        let temp = tempfile::tempdir().expect("tempdir");
        write_skill(
            temp.path(),
            "bad-skill",
            "---\nversion: \"1.0\"\n---\nBody\n",
        );

        let skill_path = temp.path().join("bad-skill");
        let result = load_skill_definition("bad-skill", &skill_path, &skill_path.join("SKILL.md"));
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("description"));
    }

    #[test]
    fn test_no_frontmatter_rejected() {
        let temp = tempfile::tempdir().expect("tempdir");
        write_skill(temp.path(), "no-fm", "Just body, no frontmatter\n");

        let skill_path = temp.path().join("no-fm");
        let result = load_skill_definition("no-fm", &skill_path, &skill_path.join("SKILL.md"));
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("frontmatter"));
    }

    #[test]
    fn test_malformed_yaml_rejected() {
        let temp = tempfile::tempdir().expect("tempdir");
        write_skill(
            temp.path(),
            "bad-yaml",
            "---\ndescription: [unclosed\n---\nBody\n",
        );

        let skill_path = temp.path().join("bad-yaml");
        let result = load_skill_definition("bad-yaml", &skill_path, &skill_path.join("SKILL.md"));
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn test_load_skill_body_with_substitution() {
        let temp = tempfile::tempdir().expect("tempdir");
        write_skill(
            temp.path(),
            "var-skill",
            "---\n\
             description: X\n\
             ---\n\
             Path: ${MEKA_SKILL_DIR}\nSession: ${MEKA_SESSION_ID}\n",
        );

        let skill_path = temp.path().join("var-skill");
        let skill = load_skill_definition("var-skill", &skill_path, &skill_path.join("SKILL.md"))
            .expect("load");

        let body = load_skill_body(&skill, Some("abc-123"))
            .await
            .expect("body");
        assert!(body.contains(&skill_path.display().to_string()));
        assert!(body.contains("Session: abc-123"));
    }

    #[tokio::test]
    async fn test_load_skill_body_without_session_id() {
        let temp = tempfile::tempdir().expect("tempdir");
        write_skill(
            temp.path(),
            "var-skill",
            "---\n\
             description: X\n\
             ---\n\
             Session: ${MEKA_SESSION_ID}\n",
        );

        let skill_path = temp.path().join("var-skill");
        let skill = load_skill_definition("var-skill", &skill_path, &skill_path.join("SKILL.md"))
            .expect("load");

        let body = load_skill_body(&skill, None).await.expect("body");
        assert!(body.contains("Session: ${MEKA_SESSION_ID}"));
    }

    fn valid_frontmatter(description: &str) -> String {
        format!("---\ndescription: {}\n---\nBody\n", description)
    }

    /// Bump the mtime of a file far enough in the future to defeat 1-second filesystem resolution.
    /// Uses `File::set_modified` (stable since Rust 1.75) so no extra dep is required.
    fn bump_mtime(path: &Path) {
        let file = std::fs::OpenOptions::new()
            .write(true)
            .open(path)
            .expect("open for mtime bump");
        let future = SystemTime::now() + std::time::Duration::from_secs(10);
        file.set_modified(future).expect("set_modified");
    }

    #[tokio::test]
    async fn test_skill_cache_picks_up_new_skill() {
        let temp = tempfile::tempdir().expect("tempdir");
        let cache = SkillCache::for_root(Some(temp.path().to_path_buf()));
        assert!(cache.current().await.is_empty());

        write_skill(temp.path(), "foo", &valid_frontmatter("first"));

        let skills = cache.current().await;
        assert_eq!(skills.len(), 1);
        assert_eq!(skills[0].name, "foo");
    }

    #[tokio::test]
    async fn test_skill_cache_detects_modified_frontmatter() {
        let temp = tempfile::tempdir().expect("tempdir");
        write_skill(temp.path(), "foo", &valid_frontmatter("old"));

        let cache = SkillCache::for_root(Some(temp.path().to_path_buf()));
        let skills = cache.current().await;
        assert_eq!(skills[0].description, "old");

        let skill_md = temp.path().join("foo").join("SKILL.md");
        std::fs::write(&skill_md, valid_frontmatter("new")).expect("rewrite");
        bump_mtime(&skill_md);

        let skills = cache.current().await;
        assert_eq!(skills[0].description, "new");
    }

    #[tokio::test]
    async fn test_skill_cache_drops_removed_skill() {
        let temp = tempfile::tempdir().expect("tempdir");
        write_skill(temp.path(), "foo", &valid_frontmatter("first"));

        let cache = SkillCache::for_root(Some(temp.path().to_path_buf()));
        assert_eq!(cache.current().await.len(), 1);

        std::fs::remove_dir_all(temp.path().join("foo")).expect("rm skill");
        let skills = cache.current().await;
        assert!(skills.is_empty());
    }

    #[tokio::test]
    async fn test_skill_cache_stable_when_unchanged() {
        let temp = tempfile::tempdir().expect("tempdir");
        write_skill(temp.path(), "foo", &valid_frontmatter("first"));

        let cache = SkillCache::for_root(Some(temp.path().to_path_buf()));
        let first = cache.current().await;
        let second = cache.current().await;

        // Same Arc pointer ⇒ no rediscovery happened, which proves the cache really did skip the
        // inner walk on the stable-snapshot path.
        assert!(
            Arc::ptr_eq(&first, &second),
            "expected cache to skip rediscovery when nothing changed"
        );
    }

    #[test]
    fn test_skill_context_header_points_at_source_dir() {
        let temp = tempfile::tempdir().expect("tempdir");
        write_skill(temp.path(), "demo", &valid_frontmatter("x"));
        let skill_path = temp.path().join("demo");
        let skill =
            load_skill_definition("demo", &skill_path, &skill_path.join("SKILL.md")).expect("load");

        let header = skill_context_header(&skill);
        assert!(header.contains("bundled files"));
        assert!(header.contains(&skill_path.display().to_string()));
    }

    #[tokio::test]
    async fn test_skill_cache_with_no_root_is_empty() {
        let cache = SkillCache::for_root(None);
        assert!(cache.current().await.is_empty());
    }
}
