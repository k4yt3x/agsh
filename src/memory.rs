//! Agent memory: durable notes the agent writes for itself, surviving compaction and outliving any
//! one session.
//!
//! Memories are rows in the `memories` table of meka's database (`MEKA_DATA_DIR`), one per note,
//! carrying a required one-line `description` and an optional body, priority and tag set. The store
//! is scoped to the meka *instance*, not to a session or a directory: meka has no Project concept,
//! and the motivating deployment is a single always-on session reachable over chat, where the agent
//! is closer to a person than to a checkout.
//!
//! This module owns the vocabulary -- what a [`Memory`] is, what names and tags are legal, how an
//! age is rendered, and how a memory is written out for `meka memory export`. Storage and retrieval
//! are [`store`], which is the source of truth.
//!
//! One row is the whole of a memory: there is no second copy to keep in step, and a transaction is
//! what makes a write atomic rather than a lock. `meka memory export` is the answer for anyone who
//! wants `$EDITOR`, `grep` or a git-able directory over the same content.
//!
//! [`crate::skills`] is deliberately not built this way. A `SKILL.md` is a shared spec format other
//! clients read, and a skill directory carries bundled scripts and assets, so files are right
//! there.
//!
//! Why this survives compaction: the index rides [`crate::context::WorldSnapshot`], which
//! `Agent::last_rendered_world` re-states in full at session start, after every compaction, and
//! whenever the previous render scrolls out of the context window.

pub mod cli;
pub mod store;

use std::{
    path::{Path, PathBuf},
    time::SystemTime,
};

pub use self::store::MemoryStore;
// Re-exported rather than referenced through `crate::store` at each use site: priority is part
// of the memory store's public vocabulary (`meka memory add --priority`, the `memory_write`
// schema), and the constants moved there only so `skills` could share the same scale.
pub use crate::store::{DEFAULT_PRIORITY, MAX_PRIORITY, MIN_PRIORITY, normalize_description};
use crate::store::{validate_entry_name, yaml_scalar};

/// A single durable note, as one row of the `memories` table.
///
/// `description` is what the agent sees every turn; `body` is fetched on demand through the
/// `memory_read` tool.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Memory {
    pub name: String,
    pub description: String,
    pub priority: u8,
    /// Free-form labels, validated to [`validate_tag`]'s character class.
    ///
    /// What makes a store of thousands navigable rather than merely searchable: the index can only
    /// show a couple of hundred entries, and "4,910 more memories not shown" is not a usable
    /// signal, where "most common tags infra, people, decisions" is a query the model can act on.
    pub tags: Vec<String>,
    /// When the memory was *recorded*: stamped once, at create, and carried across every later
    /// write by the upsert itself (see [`store::MemoryStore::write`]).
    ///
    /// Distinct from [`Self::updated_at`] because the two answer different questions and only one
    /// of them is the one being asked. A `memory_write` that reworded a description used to move
    /// the one timestamp there was, which made a years-old note render as "today", sort to the top
    /// of its priority band, and arrive through `memory_read` under the caption "Saved today. This
    /// is what you recorded then".
    pub recorded_at: SystemTime,
    /// When the row was last written. Reported by `meka memory get` and the HTTP API; it takes no
    /// part in ordering, ranking or the rendered age.
    pub updated_at: SystemTime,
    /// How many times `memory_read` has opened this memory. Feeds the usage weight in
    /// [`store::Ranking`], which is the counterweight to a priority the agent guessed once and
    /// never revised.
    pub read_count: u32,
    /// The body text, present when the query that produced this loaded it.
    ///
    /// [`store::MemoryStore::get`] always loads it; [`store::MemoryStore::index`] loads it only
    /// for the band the `[Memory]` section renders in full (see [`INLINE_BODY_PRIORITY_MAX`]),
    /// because carrying every body would put the whole store in resident memory for the sake
    /// of a handful of entries.
    pub body: Option<String>,
}

/// Priority at or below which a memory's body is rendered into the per-turn context in full,
/// rather than represented by its description.
///
/// Zero, not the 0-1 band `memory_write` calls "standing directives". For a standing rule the body
/// *is* the rule, and leaving it behind a tool call the model may never make is the gap this
/// closes -- but inlining two whole bands doubles the chance of blowing the budget and pushing the
/// index itself out, so the always-in-context tier is deliberately the narrower one.
pub const INLINE_BODY_PRIORITY_MAX: u8 = 0;

/// Most tags one memory may carry. A tag set is a handful of labels; past this it is a body.
pub const MAX_TAGS: usize = 10;
/// Longest a single tag may be. Bounded so the histogram in the `[Memory]` index stays readable.
pub const MAX_TAG_LEN: usize = 32;

/// Validate one tag: lowercase alphanumerics and hyphens, starting with an alphanumeric.
///
/// Strict deliberately, and load-bearing rather than cosmetic in two places. Tags are stored
/// space-joined in one column, so a tag containing a space would come back as two; and
/// [`render_memory`] emits the list as a YAML flow sequence through [`yaml_scalar`], which is safe
/// only for values already normalised to one line.
pub fn validate_tag(tag: &str) -> Result<(), String> {
    if tag.is_empty() {
        return Err("a tag cannot be empty".to_string());
    }
    if tag.chars().count() > MAX_TAG_LEN {
        return Err(format!("tag '{}' exceeds {} characters", tag, MAX_TAG_LEN));
    }
    let mut chars = tag.chars();
    // Non-empty was checked above, so this always yields `Some`.
    #[allow(clippy::expect_used)]
    let first = chars.next().expect("non-empty checked above");
    if !(first.is_ascii_lowercase() || first.is_ascii_digit()) {
        return Err(format!(
            "tag '{}' must start with a lowercase letter or digit",
            tag
        ));
    }
    for character in chars {
        if !(character.is_ascii_lowercase() || character.is_ascii_digit() || character == '-') {
            return Err(format!(
                "tag '{}' contains invalid character '{}'; only [a-z0-9-] are allowed",
                tag, character
            ));
        }
    }
    Ok(())
}

/// Validate a whole tag list for a write, rejecting the set rather than silently dropping members.
pub fn validate_tags(tags: &[String]) -> Result<(), String> {
    if tags.len() > MAX_TAGS {
        return Err(format!(
            "{} tags given; at most {} are allowed",
            tags.len(),
            MAX_TAGS
        ));
    }
    for tag in tags {
        validate_tag(tag)?;
    }
    Ok(())
}

/// Lowercase, sort, deduplicate and validate a tag list on its way to the store.
///
/// Lowercased *before* validating, so the doors agree: search normalises case anyway, and refusing
/// `Infra` here would mean a label meka itself renders is one meka itself will not take back.
/// Sorted before the dedup because `dedup` only removes *consecutive* duplicates, so the other
/// order leaves `[a, b, a]` intact and the row ends up declaring `a` twice.
pub fn normalize_tags(tags: &[String]) -> Result<Vec<String>, String> {
    let mut tags: Vec<String> = tags.iter().map(|tag| tag.trim().to_lowercase()).collect();
    tags.sort();
    tags.dedup();
    validate_tags(&tags)?;
    Ok(tags)
}

/// Parse an RFC 3339 timestamp column, or `None` if it is not one.
pub(crate) fn parse_recorded_str(raw: &str) -> Option<SystemTime> {
    chrono::DateTime::parse_from_rfc3339(raw)
        .ok()
        .map(|parsed| SystemTime::from(parsed.with_timezone(&chrono::Utc)))
}

/// Render a [`SystemTime`] as the RFC 3339 string the timestamp columns carry.
pub(crate) fn render_recorded(time: SystemTime) -> String {
    chrono::DateTime::<chrono::Utc>::from(time).to_rfc3339()
}

/// The file name one exported memory lands under. The sole owner of the `<name>.md` layout
/// convention, so changing it is a one-line edit rather than a grep.
fn memory_file_name(name: &str) -> String {
    format!("{}.md", name)
}

/// Resolve one exported memory's path inside `root`. Performs no I/O and does not validate the
/// name; callers pair it with [`validate_memory_name`].
pub fn memory_file_in(root: &Path, name: &str) -> PathBuf {
    root.join(memory_file_name(name))
}

/// Validate that `name` is a safe prompt-embeddable memory identifier. See [`validate_entry_name`]
/// for the rules.
///
/// Still load-bearing after the move off the filesystem, for two reasons that outlived it: a name
/// is what `meka memory export` turns into a file name, and it is text the model reads in every
/// turn's index.
pub fn validate_memory_name(name: &str) -> Result<(), String> {
    validate_entry_name(name, "memory")
}

/// Bound a name being *looked up*, without demanding it be one this store would write.
///
/// The lookup half of [`validate_memory_name`]; see [`crate::store::validate_lookup_name`] for why
/// a store that applies its write rule to reads and deletes cannot get rid of a row it should never
/// have accepted.
pub fn validate_memory_lookup(name: &str) -> Result<(), String> {
    crate::store::validate_lookup_name(name, "memory")
}

/// Clamp a caller-supplied `priority` for a memory. Thin wrapper over
/// [`crate::store::parse_priority`] that supplies this store's noun, mirroring
/// [`validate_memory_name`].
pub fn parse_priority(raw: Option<i64>, name: &str) -> u8 {
    crate::store::parse_priority(raw, "memory", name)
}

/// The header an exported memory file carries.
///
/// A struct rather than four positional arguments, mirroring the same simplification
/// `render_skill_file` received: `render_memory("x", 5, None, &[], body)` is unreadable at the call
/// site and silently accepts a swapped pair.
#[derive(Debug, Clone)]
pub struct MemoryFrontmatter {
    pub description: String,
    pub priority: u8,
    /// The `recorded:` value as it will appear in the file, RFC 3339.
    pub recorded: Option<String>,
    pub tags: Vec<String>,
    /// How many times the memory has been read, emitted only when non-zero.
    ///
    /// Not content but usage: what the agent has *done* with the note. It rides along in an export
    /// because it is the one value the rest of the file cannot reconstruct -- descriptions, bodies
    /// and dates are all there, but a store restored with every counter at zero has silently lost
    /// each memory's accumulated ranking weight. A reader that does not model the key ignores it,
    /// as it ignores any other it does not model.
    pub read_count: u32,
}

/// Render one memory as a Markdown file: frontmatter followed by the body.
///
/// The export format, and the only place meka writes YAML for this store. Frontmatter rather than
/// JSON because the point of an export is to be read and edited by a person, and to be greppable
/// in a directory the way the rest of a notes tree is.
///
/// `priority` is emitted only when it differs from [`DEFAULT_PRIORITY`], `recorded` only when
/// known, and `tags` only when non-empty, so the common case stays a two-line header.
pub fn render_memory(frontmatter: &MemoryFrontmatter, body: &str) -> String {
    use std::fmt::Write as _;

    let mut out = String::new();
    out.push_str("---\n");
    // Only the characters YAML genuinely cannot carry are dropped. A C0 or C1 control inside a
    // double-quoted scalar is outside YAML's `c-printable` production, so an export holding one is
    // a file no parser will read. `sanitize_stored_description` was used here at first and went
    // further than that argument: it also strips the whole `Cf` category, so a Persian description
    // came back a different word after a backup -- a read that sanitises, written to a persistent
    // store, which is the class this whole change closed for bodies.
    let _ = writeln!(
        out,
        "description: {}",
        yaml_scalar(&normalize_description(&yaml_printable(
            &frontmatter.description
        )))
    );
    if frontmatter.priority != DEFAULT_PRIORITY {
        let _ = writeln!(out, "priority: {}", frontmatter.priority);
    }
    if let Some(recorded) = &frontmatter.recorded {
        // Quoted by `yaml_scalar` on the strength of the colons in the time, which is what keeps
        // the offset from parsing as a nested mapping.
        let _ = writeln!(out, "recorded: {}", yaml_scalar(recorded));
    }
    if frontmatter.read_count > 0 {
        let _ = writeln!(out, "read_count: {}", frontmatter.read_count);
    }
    if !frontmatter.tags.is_empty() {
        // Each element through `yaml_scalar` rather than a bare flow sequence. Every tag reaching
        // here has passed `validate_tag`, so the bare form would be safe today; quoting costs
        // nothing and means a later loosening of that character class cannot quietly produce a file
        // the importer reads as something else.
        let quoted: Vec<String> = frontmatter
            .tags
            .iter()
            .map(|tag| yaml_scalar(tag))
            .collect();
        let _ = writeln!(out, "tags: [{}]", quoted.join(", "));
    }
    // The body verbatim between one separator newline and one terminator newline, both added
    // unconditionally. Trimming leading newlines and appending a terminator only when one was
    // missing made the framing ambiguous, so the importer could not tell padding from content: a
    // body of `b` came back `\nb`, and one ending `\r` came back ending `\r\n`. Adding exactly one
    // of each, always, is what makes the round trip exact for every body including an empty one.
    out.push_str("---\n\n");
    out.push_str(body);
    out.push('\n');
    out
}

/// Drop the characters YAML cannot represent in a scalar, and nothing else.
///
/// Deliberately narrower than [`render_for_model`]: this is the *export* path, so the only
/// justification for touching the text is that the file would otherwise be unparseable. Format
/// characters (ZWJ, ZWNJ, bidi marks) are `c-printable` and stay.
fn yaml_printable(text: &str) -> String {
    text.chars()
        .filter(|character| {
            let code = *character as u32;
            *character == '\n'
                || *character == '\t'
                || (code >= 0x20 && code != 0x7F && !(0x80..=0x9F).contains(&code))
        })
        .collect()
}

/// Whether a description will still say something by the time the model reads it.
///
/// Every write door asked `trim().is_empty()`, which is a question about whitespace. Format
/// characters -- zero-width spaces, joiners, bidi controls -- are not whitespace, so three of them
/// passed as a description and then rendered as nothing: a blank cell in `meka memory list`, a
/// blank line in `memory_search`, and `- **name** (p5, today): ` in the index the model reads every
/// turn. [`render_description_for_model`] strips exactly that class at the render boundary, so
/// asking it is asking the question the store actually has.
///
/// Distinct from [`description_survives_export`], which asks whether YAML can carry the text. A
/// description can fail either check independently.
pub fn description_says_something(description: &str) -> bool {
    !render_description_for_model(description).trim().is_empty()
}

/// Whether a description would still say something once written to an export file.
///
/// [`render_memory`] drops what YAML cannot carry, and a description made only of such characters
/// becomes `description: ""` -- which the importer treats as no description at all and skips,
/// losing the memory through the one path that is supposed to preserve it. `meka memory export`
/// asks this before it writes anything, and refuses the whole run rather than write a file that
/// would come back empty.
///
/// Distinct from [`description_says_something`], which asks whether the *model* would see anything.
/// A description can fail either check independently: YAML carries a zero-width space fine, and the
/// render boundary strips it.
pub fn description_survives_export(description: &str) -> bool {
    !normalize_description(&yaml_printable(description))
        .trim()
        .is_empty()
}

/// Render one [`Memory`] as an export file, body included.
pub fn export_memory(memory: &Memory) -> String {
    render_memory(
        &MemoryFrontmatter {
            description: memory.description.clone(),
            priority: memory.priority,
            recorded: Some(render_recorded(memory.recorded_at)),
            tags: memory.tags.clone(),
            read_count: memory.read_count,
        },
        memory.body.as_deref().unwrap_or_default(),
    )
}

/// Make stored memory text safe to render into a model's context or a terminal.
///
/// The store returns bytes (see `store::row_to_memory`), because `meka memory edit` round-trips a
/// body through `$EDITOR` and a read that stripped characters would destroy them permanently.
/// Neutralising therefore happens here, at each boundary where the text is *displayed* rather than
/// carried: the `[Memory]` index and its standing band, `memory_read`, both search renderers, a
/// sub-agent's index, and the `meka memory` listing.
///
/// The two doors that deliberately do **not** call this are `meka memory export` and
/// `meka memory edit`, which exist to hand back exactly what is stored.
pub fn render_for_model(text: &str) -> String {
    crate::mcp::sanitize::sanitize_text(text)
}

/// The same for a description, which is additionally contracted to be one line.
///
/// A memory whose description carries a newline would otherwise open what looks like a new section
/// in the per-turn index the model reads every turn.
pub fn render_description_for_model(description: &str) -> String {
    crate::store::sanitize_stored_description(description)
}

/// Human-readable age, e.g. "today", "yesterday", "47 days ago".
///
/// Deliberately not an ISO timestamp: models are poor at date arithmetic, and a rendered age
/// prompts staleness reasoning in a way a raw date does not. A memory is a point-in-time
/// observation, and the agent needs to weigh an old one accordingly.
///
/// Callers pass [`Memory::recorded_at`], never [`Memory::updated_at`]. The number is only worth
/// rendering if it answers "how old is what this says"; an edit date dressed up as an observation
/// date is worse than no date, because it reads as a fact the model can rely on.
pub fn render_age(recorded: SystemTime, now: SystemTime) -> String {
    // A stamp in the future is its own answer, not "today". `duration_since` fails for one, and
    // folding that to zero told the model a note dated next year had been written this morning --
    // while the same row sorts to the top of its priority band, so the memory most likely to be
    // wrong is also the most prominent. Reachable through clock skew between two machines sharing a
    // data directory, or a hand-written date. Saying so is what lets the model discount it.
    let Ok(elapsed) = now.duration_since(recorded) else {
        return "at a future date".to_string();
    };
    match elapsed.as_secs() / 86_400 {
        0 => "today".to_string(),
        1 => "yesterday".to_string(),
        n => format!("{} days ago", n),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_validate_memory_name_rejects_traversal_and_separators() {
        for bad in [
            "../escape",
            "a/b",
            "a\\b",
            "/abs",
            ".hidden",
            "",
            "with space",
        ] {
            assert!(
                validate_memory_name(bad).is_err(),
                "'{}' must be rejected",
                bad
            );
        }
        for good in ["alice-timezone", "deploy_host", "note42", "A-B_c9"] {
            assert!(
                validate_memory_name(good).is_ok(),
                "'{}' must be accepted",
                good
            );
        }
    }

    #[test]
    fn test_priority_defaults_and_clamps() {
        assert_eq!(parse_priority(None, "n"), DEFAULT_PRIORITY);
        assert_eq!(parse_priority(Some(0), "n"), 0);
        assert_eq!(parse_priority(Some(9), "n"), 9);
        assert_eq!(parse_priority(Some(-5), "n"), MIN_PRIORITY);
        assert_eq!(parse_priority(Some(99), "n"), MAX_PRIORITY);
    }

    /// A tag is stored space-joined in one column and rendered into a YAML flow sequence, so
    /// anything either of those would read as structure has to be refused at the door.
    #[test]
    fn test_tag_validation_refuses_anything_yaml_would_interpret() {
        for bad in [
            "",
            "Infra",
            "-leading",
            "has space",
            "with:colon",
            "under_score",
            "quote\"mark",
            "comma,tag",
            "bracket]",
            &"x".repeat(MAX_TAG_LEN + 1),
        ] {
            assert!(validate_tag(bad).is_err(), "'{}' must be rejected", bad);
        }
        for good in ["infra", "deploy-host", "k8s", "0day"] {
            assert!(validate_tag(good).is_ok(), "'{}' must be accepted", good);
        }
        assert!(validate_tags(&vec!["a".to_string(); MAX_TAGS]).is_ok());
        assert!(validate_tags(&vec!["a".to_string(); MAX_TAGS + 1]).is_err());
    }

    /// The normalisation the write door applies. Sorting before the dedup is the load-bearing part:
    /// `dedup` only removes *consecutive* duplicates, so `[a, b, a]` would otherwise survive whole
    /// and the row would declare `a` twice.
    #[test]
    fn test_normalize_tags_lowercases_sorts_and_deduplicates() {
        assert_eq!(
            normalize_tags(&[
                "Infra".to_string(),
                "deploy".to_string(),
                " infra ".to_string()
            ])
            .expect("valid"),
            vec!["deploy".to_string(), "infra".to_string()]
        );
        assert!(normalize_tags(&["has space".to_string()]).is_err());
    }

    /// An export has to parse back as YAML frontmatter, or `meka memory export` produces files
    /// that only look like the format they claim.
    #[test]
    fn test_export_round_trips_through_frontmatter() {
        let memory = Memory {
            name: "note".to_string(),
            description: "a description: with a colon".to_string(),
            priority: 2,
            tags: vec!["deploy".to_string(), "infra".to_string()],
            recorded_at: SystemTime::UNIX_EPOCH + std::time::Duration::from_secs(1_700_000_000),
            updated_at: SystemTime::now(),
            read_count: 3,
            body: Some("Body text.\n".to_string()),
        };
        let rendered = export_memory(&memory);
        let (frontmatter, body) =
            crate::store::split_frontmatter(&rendered).expect("must have frontmatter");
        let parsed: serde_norway::Value =
            serde_norway::from_str(frontmatter).expect("frontmatter must parse");
        assert_eq!(
            parsed["description"].as_str(),
            Some("a description: with a colon")
        );
        assert_eq!(parsed["priority"].as_i64(), Some(2));
        assert_eq!(
            parsed["recorded"].as_str(),
            Some("2023-11-14T22:13:20+00:00")
        );
        assert_eq!(
            parsed["tags"]
                .as_sequence()
                .map(|tags| tags.len())
                .unwrap_or(0),
            2
        );
        // The one value the file cannot otherwise carry: what the agent has done with the note.
        assert_eq!(parsed["read_count"].as_i64(), Some(3));
        assert_eq!(body.trim(), "Body text.");
    }

    /// A default priority and an empty tag set are omitted, so an export stays readable.
    #[test]
    fn test_export_omits_default_priority_and_empty_tags() {
        let memory = Memory {
            name: "note".to_string(),
            description: "plain".to_string(),
            priority: DEFAULT_PRIORITY,
            tags: Vec::new(),
            recorded_at: SystemTime::UNIX_EPOCH,
            updated_at: SystemTime::UNIX_EPOCH,
            read_count: 0,
            body: None,
        };
        let rendered = export_memory(&memory);
        assert!(!rendered.contains("priority:"), "{rendered}");
        assert!(!rendered.contains("tags:"), "{rendered}");
        assert!(!rendered.contains("read_count:"), "{rendered}");
        assert!(rendered.contains("recorded:"), "{rendered}");
    }

    #[test]
    fn test_render_age() {
        let now = SystemTime::now();
        assert_eq!(render_age(now, now), "today");
        assert_eq!(
            render_age(now - std::time::Duration::from_secs(86_400), now),
            "yesterday"
        );
        assert_eq!(
            render_age(now - std::time::Duration::from_secs(47 * 86_400), now),
            "47 days ago"
        );
        // A stamp in the future must not panic, and must not read as "today" either: such a row
        // sorts to the top of its priority band, so the memory most likely to be wrong would also
        // be the most prominent, with nothing saying why.
        assert_eq!(
            render_age(now + std::time::Duration::from_secs(86_400), now),
            "at a future date"
        );
    }
}
