//! Shared plumbing for the two on-disk Markdown stores meka owns: skills
//! (`~/.config/meka/skills/<name>/SKILL.md`, [`crate::skills`]) and memories
//! (`~/.config/meka/memory/<name>.md`, [`crate::memory`]).
//!
//! Both use the same `---`-fenced YAML header, the same quoting rules when meka writes one back
//! out, and the same rules for what makes a legal entry name, so those three pieces live here
//! rather than being reimplemented per store.

/// Split a file into (frontmatter, body) if it starts with a `---` fence. Returns None when no
/// valid frontmatter block is present.
pub(crate) fn split_frontmatter(content: &str) -> Option<(&str, &str)> {
    let rest = content
        .strip_prefix("---\n")
        .or_else(|| content.strip_prefix("---\r\n"))?;

    for (marker, offset) in [("\n---\n", 5), ("\n---\r\n", 6)] {
        if let Some(end) = rest.find(marker) {
            let frontmatter = &rest[..end];
            let body = &rest[end + offset..];
            return Some((frontmatter, body));
        }
    }
    None
}

/// YAML-quote a scalar when it contains characters that would otherwise require structural
/// interpretation. Plain ASCII text without leading punctuation, colons, or hash marks passes
/// through unquoted.
pub(crate) fn yaml_scalar(text: &str) -> String {
    let needs_quotes = text.is_empty()
        || text.starts_with([
            '-', '?', ':', '!', '&', '*', '#', '|', '>', '%', '@', '`', '"', '\'',
        ])
        || text.contains(':')
        || text.contains('#')
        || text.contains('\n');
    if needs_quotes {
        let escaped = text.replace('\\', "\\\\").replace('"', "\\\"");
        format!("\"{}\"", escaped)
    } else {
        text.to_string()
    }
}

/// Maximum length of a store entry's name. Bounded so an index line in the per-turn context stays
/// readable and per-line bounded.
pub(crate) const MAX_ENTRY_NAME_LEN: usize = 64;

/// Validate that `name` is a safe filesystem-and-prompt-embeddable identifier for a store entry:
/// `[A-Za-z0-9][A-Za-z0-9_-]*`, at most [`MAX_ENTRY_NAME_LEN`] characters. `noun` names the store
/// in the error text ("skill", "memory").
///
/// Rejecting everything outside the character class rules out `..`, path separators, absolute
/// paths, and dot-files *by construction* rather than by enumerating the attacks. That matters most
/// for memory, whose tools run at [`crate::permission::Permission`] `Read`: without this check
/// `memory_write` would be an arbitrary-file-write primitive reachable in read-only mode.
pub(crate) fn validate_entry_name(name: &str, noun: &str) -> Result<(), String> {
    if name.is_empty() {
        return Err(format!("{} name cannot be empty", noun));
    }
    if name.len() > MAX_ENTRY_NAME_LEN {
        return Err(format!(
            "{} name '{}' exceeds {} characters",
            noun, name, MAX_ENTRY_NAME_LEN
        ));
    }
    let mut chars = name.chars();
    // `name.is_empty()` was checked above (and returned an error), so this always yields `Some`.
    // The `expect` documents the invariant.
    #[allow(clippy::expect_used)]
    let first = chars.next().expect("non-empty checked above");
    if !first.is_ascii_alphanumeric() {
        return Err(format!(
            "{} name '{}' must start with a letter or digit",
            noun, name
        ));
    }
    for ch in chars {
        if !(ch.is_ascii_alphanumeric() || ch == '_' || ch == '-') {
            return Err(format!(
                "{} name '{}' contains invalid character '{}'; only [A-Za-z0-9_-] are allowed",
                noun, name, ch
            ));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_split_frontmatter_simple() {
        let content = "---\ndescription: hi\n---\nbody here\n";
        let (fm, body) = split_frontmatter(content).expect("should split");
        assert_eq!(fm, "description: hi");
        assert_eq!(body, "body here\n");
    }

    #[test]
    fn test_split_frontmatter_crlf() {
        let content = "---\r\ndescription: hi\r\n---\r\nbody\r\n";
        let split = split_frontmatter(content);
        assert!(split.is_some());
    }

    #[test]
    fn test_split_frontmatter_no_fence() {
        let content = "no frontmatter here\n";
        assert!(split_frontmatter(content).is_none());
    }

    #[test]
    fn test_yaml_scalar_plain_text_is_unquoted() {
        assert_eq!(yaml_scalar("A plain description"), "A plain description");
    }

    /// A description containing a colon is the common case that forces quoting: unquoted it would
    /// parse as a nested mapping and the whole frontmatter block would be rejected.
    #[test]
    fn test_yaml_scalar_quotes_structural_characters() {
        assert_eq!(yaml_scalar("note: with colon"), "\"note: with colon\"");
        assert_eq!(yaml_scalar("- leading dash"), "\"- leading dash\"");
        assert_eq!(yaml_scalar("has # hash"), "\"has # hash\"");
        assert_eq!(yaml_scalar(""), "\"\"");
    }

    #[test]
    fn test_yaml_scalar_escapes_quotes_and_backslashes() {
        assert_eq!(yaml_scalar("say \"hi\": now"), "\"say \\\"hi\\\": now\"");
        assert_eq!(yaml_scalar("back\\slash: x"), "\"back\\\\slash: x\"");
    }

    /// The character class is the security boundary for `memory_write`, which runs at read
    /// permission, so these must be rejected by construction rather than by special case.
    #[test]
    fn test_validate_entry_name_rejects_escapes() {
        for bad in [
            "",
            "..",
            "../escape",
            "a/b",
            "a\\b",
            "/abs",
            ".hidden",
            "-lead",
            "has space",
            "has:colon",
        ] {
            assert!(
                validate_entry_name(bad, "memory").is_err(),
                "'{bad}' must be rejected"
            );
        }
        for good in ["a", "note", "a-note_2", "K4YT3X-prefers-terse"] {
            assert!(
                validate_entry_name(good, "skill").is_ok(),
                "'{good}' must be accepted"
            );
        }
        assert!(validate_entry_name(&"a".repeat(MAX_ENTRY_NAME_LEN + 1), "skill").is_err());
    }

    #[test]
    fn test_validate_entry_name_uses_the_caller_s_noun() {
        let error = validate_entry_name("", "memory").expect_err("empty is invalid");
        assert!(error.starts_with("memory name"), "{error}");
        let error = validate_entry_name("", "skill").expect_err("empty is invalid");
        assert!(error.starts_with("skill name"), "{error}");
    }
}
