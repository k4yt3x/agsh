//! String hygiene helpers for data crossing the MCP boundary. Strips invisible/control characters
//! that could be used for prompt injection or UI spoofing (RTL/LTR overrides, zero-width joiners,
//! C0/C1 controls) and normalises MCP server names to the alphabet accepted by provider tool
//! schemas.

/// Reserved server names that collide with meka internals or with the tool namespace separator.
/// Connection requests for these names are rejected.
pub const RESERVED_SERVER_NAMES: &[&str] = &["meka", "ide"];

/// Strip control + format characters that could hijack the terminal or be
/// used as homograph-style attacks on users reviewing tool output:
///
/// - Unicode category **Cc** (C0/C1 controls) except `\n`, `\t`, `\r`.
/// - Unicode category **Cf** (formatters: RTL/LTR overrides, zero-width joiners, byte-order marks,
///   language tags).
/// - Unpaired surrogate code units (already impossible in a valid `&str`, noted for completeness).
///
/// Emoji, CJK, combining marks, and all other printable Unicode pass through unchanged.
pub fn sanitize_text(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    for ch in input.chars() {
        if is_safe_char(ch) {
            out.push(ch);
        }
    }
    out
}

fn is_safe_char(ch: char) -> bool {
    // Whitelist the three whitespace controls we care about.
    if ch == '\n' || ch == '\t' || ch == '\r' {
        return true;
    }
    let code = ch as u32;

    // C0 controls (U+0000–U+001F) and DEL (U+007F).
    if code < 0x20 || code == 0x7F {
        return false;
    }

    // C1 controls (U+0080–U+009F).
    if (0x80..=0x9F).contains(&code) {
        return false;
    }

    // Cf category (Format): covers RTL/LTR overrides, ZWJ/ZWNJ, BOM, interlinear annotations, and
    // language tags (E0000–E007F).
    if is_format_char(code) {
        return false;
    }

    true
}

/// Returns true for Unicode General Category `Cf` (Format).
///
/// Enumerated from Unicode 15.1: only the ranges that exist; the bulk of the BMP has no Cf
/// characters so this stays a short list.
pub(crate) fn is_format_char(code: u32) -> bool {
    matches!(
        code,
        0x00AD                   // SOFT HYPHEN
        | 0x0600..=0x0605        // Arabic number signs
        | 0x061C                 // ARABIC LETTER MARK
        | 0x06DD                 // ARABIC END OF AYAH
        | 0x070F                 // SYRIAC ABBREVIATION MARK
        | 0x0890..=0x0891        // Arabic POUND/PIASTRE
        | 0x08E2                 // ARABIC DISPUTED END OF AYAH
        | 0x180E                 // MONGOLIAN VOWEL SEPARATOR
        | 0x200B..=0x200F        // ZWSP, ZWNJ, ZWJ, LRM, RLM
        | 0x202A..=0x202E        // LRE, RLE, PDF, LRO, RLO
        | 0x2060..=0x2064        // WJ + invisible operators
        | 0x2066..=0x2069        // LRI, RLI, FSI, PDI
        | 0xFEFF                 // BOM / ZWNBSP
        | 0xFFF9..=0xFFFB        // Interlinear annotation anchors
        | 0x110BD                // KAITHI NUMBER SIGN
        | 0x110CD                // KAITHI NUMBER SIGN ABOVE
        | 0x13430..=0x13438      // Egyptian hieroglyph format controls
        | 0x1BCA0..=0x1BCA3      // Shorthand format controls
        | 0x1D173..=0x1D17A      // Musical symbol format controls
        | 0xE0001                // LANGUAGE TAG
        | 0xE0020..=0xE007F      // TAG characters
    )
}

/// Normalise a user-supplied MCP server name into the alphabet accepted as the `<server>` segment
/// of a `mcp__<server>__<tool>` tool name.
///
/// Any character outside `[A-Za-z0-9_-]` is replaced with `_`; runs of `_` are collapsed, and
/// leading/trailing `_` are trimmed. Empty results are mapped to `"mcp_server"` (the caller is
/// expected to reject that via the reserved check anyway, but this keeps the string non-empty).
pub fn normalize_server_name(input: &str) -> String {
    let mut out = String::with_capacity(input.len());
    let mut last_underscore = false;
    for ch in input.chars() {
        let mapped = if ch.is_ascii_alphanumeric() || ch == '_' || ch == '-' {
            ch
        } else {
            '_'
        };
        if mapped == '_' {
            if last_underscore {
                continue;
            }
            last_underscore = true;
        } else {
            last_underscore = false;
        }
        out.push(mapped);
    }
    let trimmed = out.trim_matches('_').to_string();
    if trimmed.is_empty() {
        "mcp_server".to_string()
    } else {
        trimmed
    }
}

/// Returns true when the given name is reserved (either an explicit entry in
/// [`RESERVED_SERVER_NAMES`] or starts with the `mcp_` prefix that meka uses for internal tools).
pub fn is_reserved_server_name(name: &str) -> bool {
    if RESERVED_SERVER_NAMES
        .iter()
        .any(|r| r.eq_ignore_ascii_case(name))
    {
        return true;
    }
    name.to_ascii_lowercase().starts_with("mcp_")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sanitize_strips_rtl_override() {
        let input = "ok\u{202E}reversed";
        assert_eq!(sanitize_text(input), "okreversed");
    }

    #[test]
    fn sanitize_strips_zero_width_space() {
        let input = "a\u{200B}b\u{200C}c\u{200D}d";
        assert_eq!(sanitize_text(input), "abcd");
    }

    #[test]
    fn sanitize_strips_bom() {
        let input = "\u{FEFF}hello";
        assert_eq!(sanitize_text(input), "hello");
    }

    #[test]
    fn sanitize_keeps_emoji() {
        let input = "hi 🦀 from rust";
        assert_eq!(sanitize_text(input), "hi 🦀 from rust");
    }

    #[test]
    fn sanitize_keeps_cjk() {
        let input = "日本語と한국어";
        assert_eq!(sanitize_text(input), "日本語と한국어");
    }

    #[test]
    fn sanitize_keeps_newlines_and_tabs() {
        let input = "line1\nline2\tcol";
        assert_eq!(sanitize_text(input), "line1\nline2\tcol");
    }

    #[test]
    fn sanitize_strips_c0_control() {
        let input = "a\x01b\x08c";
        assert_eq!(sanitize_text(input), "abc");
    }

    #[test]
    fn sanitize_strips_c1_control() {
        let input = "a\u{0085}b"; // NEL
        assert_eq!(sanitize_text(input), "ab");
    }

    #[test]
    fn normalize_simple() {
        assert_eq!(normalize_server_name("postgres"), "postgres");
    }

    #[test]
    fn normalize_replaces_invalid() {
        assert_eq!(normalize_server_name("my.server"), "my_server");
        assert_eq!(normalize_server_name("a b c"), "a_b_c");
    }

    #[test]
    fn normalize_collapses_runs() {
        assert_eq!(normalize_server_name("a...b"), "a_b");
    }

    #[test]
    fn normalize_trims_underscores() {
        assert_eq!(normalize_server_name("...foo..."), "foo");
    }

    #[test]
    fn normalize_empty_result() {
        assert_eq!(normalize_server_name("..."), "mcp_server");
    }

    #[test]
    fn normalize_keeps_dashes() {
        assert_eq!(normalize_server_name("my-server-1"), "my-server-1");
    }

    #[test]
    fn reserved_names_blocked() {
        assert!(is_reserved_server_name("meka"));
        assert!(is_reserved_server_name("IDE"));
        assert!(is_reserved_server_name("mcp_resources"));
        assert!(!is_reserved_server_name("postgres"));
    }
}
