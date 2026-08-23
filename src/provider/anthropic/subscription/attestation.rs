//! Claude Code anti-cheat machinery: request fingerprint, xxHash64-based `cch` attestation, billing
//! header synthesis, and Stainless-SDK-matching HTTP headers. All of this is OAuth-specific.
//! Direct API-key requests (`anthropic-messages`) don't send billing headers, so there's no caller.
//!
//! Every constant here is a fact about Claude Code's wire, recovered from the shipped binary rather
//! than guessed, and each is re-checked against a fresh capture when the pinned version moves. To
//! recapture, point the real client at a logging reverse proxy and set
//! `_CLAUDE_CODE_ASSUME_FIRST_PARTY_BASE_URL=1`, which is what re-enables the attestation through
//! a host that is not `api.anthropic.com`; then diff one of its requests against one of meka's.
//!
//! Claude Code ships minified, so nothing here cites a line in its source. What it cites instead is
//! the minified symbol a reader can actually grep the binary for.

use sha2::{Digest, Sha256};
use uuid::Uuid;

use crate::{
    error::{MekaError, Result},
    provider::{ContentBlock, Message, Role},
};

/// Claude Code version string. Single source of truth defined in `build.rs`.
pub(super) const CC_VERSION: &str = env!("CC_VERSION");

/// Fingerprint salt. `MHE` in the shipped 2.1.241 binary.
const FINGERPRINT_SALT: &str = "59cf53e54c78";

/// `SHA256(SALT + msg[4] + msg[7] + msg[20] + version)[:3]`.
fn compute_fingerprint(message_text: &str, version: &str) -> String {
    let indices = [4, 7, 20];
    let chars: String = indices
        .iter()
        .map(|&index| message_text.chars().nth(index).unwrap_or('0'))
        .collect();

    let input = format!("{}{}{}", FINGERPRINT_SALT, chars, version);
    let hash = Sha256::digest(input.as_bytes());
    // Match Claude Code's `SHA256(...)[:3]`: take the first 3 hex chars of the byte-by-byte
    // 2-digit-hex encoding. Two bytes give us 4 chars, enough to slice 3 and drop the rest.
    let hex: String = hash
        .iter()
        .take(2)
        .map(|byte| format!("{:02x}", byte))
        .collect();
    hex[..3].to_string()
}

/// Extracts the text content of the first user message.
fn extract_first_user_message_text(messages: &[Message]) -> String {
    for message in messages {
        if message.role == Role::User {
            for block in &message.content {
                if let ContentBlock::Text { text } = block {
                    return text.clone();
                }
            }
        }
    }
    String::new()
}

/// Computes the fingerprint from the first user message. Matches Claude
/// Code's `Kph`, which reads `$HE` (the first non-meta user message) and hashes it with `zzl`:
/// the fingerprint varies per conversation but is stable across all turns
/// of the same conversation since the first user message text doesn't
/// change.
fn compute_fingerprint_from_messages(messages: &[Message]) -> String {
    let first_message_text = extract_first_user_message_text(messages);
    compute_fingerprint(&first_message_text, CC_VERSION)
}

/// Generates the billing header with a `cch=00000` placeholder. The 3-char fingerprint suffix is
/// derived from the first user message per Claude Code's behaviour. The `cch` is replaced with the
/// real attestation by [`patch_request_body`] after serialization.
///
/// The optional segments follow in the order Claude Code's builder emits them (2.1.241, verified
/// against a wire capture): `cch`, then `cc_workload`, `cc_is_subagent`, `cc_prev_req`,
/// `cc_prompt_id`. meka never has a workload, so that one is always absent; the rest appear
/// exactly when their source does.
pub(super) fn generate_billing_header(messages: &[Message]) -> String {
    let subagent = if crate::provider::is_subagent() {
        " cc_is_subagent=true;"
    } else {
        ""
    };
    let previous_request = crate::provider::previous_request_id()
        .map(|id| format!(" cc_prev_req={};", id))
        .unwrap_or_default();
    let prompt = crate::provider::current_prompt_id()
        .map(|id| format!(" cc_prompt_id={};", id))
        .unwrap_or_default();
    format!(
        "x-anthropic-billing-header: cc_version={}.{}; cc_entrypoint=cli; cch=00000;{}{}{}",
        CC_VERSION,
        compute_fingerprint_from_messages(messages),
        subagent,
        previous_request,
        prompt,
    )
}

// xxHash64 and the `cch` attestation token. The seed and the preimage filter below are the whole
// of it; the module header says how they were recovered.

const XXH64_PRIME1: u64 = 0x9e3779b185ebca87;
const XXH64_PRIME2: u64 = 0xc2b2ae3d27d4eb4f;
const XXH64_PRIME3: u64 = 0x165667b19e3779f9;
const XXH64_PRIME4: u64 = 0x85ebca77c2b2ae63;
const XXH64_PRIME5: u64 = 0x27d4eb2f165667c5;

/// xxHash64 seed for the `cch` attestation token.
const CCH_XXH64_SEED: u64 = 0x4d659218e32a3268;

fn xxh64_round(acc: u64, lane: u64) -> u64 {
    acc.wrapping_add(lane.wrapping_mul(XXH64_PRIME2))
        .rotate_left(31)
        .wrapping_mul(XXH64_PRIME1)
}

fn xxh64_merge_round(acc: u64, val: u64) -> u64 {
    (acc ^ xxh64_round(0, val))
        .wrapping_mul(XXH64_PRIME1)
        .wrapping_add(XXH64_PRIME4)
}

fn xxh64_avalanche(mut h: u64) -> u64 {
    h ^= h >> 33;
    h = h.wrapping_mul(XXH64_PRIME2);
    h ^= h >> 29;
    h = h.wrapping_mul(XXH64_PRIME3);
    h ^= h >> 32;
    h
}

fn read_u32_le(buf: &[u8], offset: usize) -> u64 {
    u32::from_le_bytes([
        buf[offset],
        buf[offset + 1],
        buf[offset + 2],
        buf[offset + 3],
    ]) as u64
}

fn read_u64_le(buf: &[u8], offset: usize) -> u64 {
    u64::from_le_bytes([
        buf[offset],
        buf[offset + 1],
        buf[offset + 2],
        buf[offset + 3],
        buf[offset + 4],
        buf[offset + 5],
        buf[offset + 6],
        buf[offset + 7],
    ])
}

fn xxh64(input: &[u8], seed: u64) -> u64 {
    let len = input.len();
    let mut p = 0usize;
    let mut h64: u64;

    if len >= 32 {
        let mut v1 = seed.wrapping_add(XXH64_PRIME1).wrapping_add(XXH64_PRIME2);
        let mut v2 = seed.wrapping_add(XXH64_PRIME2);
        let mut v3 = seed;
        let mut v4 = seed.wrapping_sub(XXH64_PRIME1);

        let limit = len - 32;
        while p <= limit {
            v1 = xxh64_round(v1, read_u64_le(input, p));
            p += 8;
            v2 = xxh64_round(v2, read_u64_le(input, p));
            p += 8;
            v3 = xxh64_round(v3, read_u64_le(input, p));
            p += 8;
            v4 = xxh64_round(v4, read_u64_le(input, p));
            p += 8;
        }

        h64 = v1
            .rotate_left(1)
            .wrapping_add(v2.rotate_left(7))
            .wrapping_add(v3.rotate_left(12))
            .wrapping_add(v4.rotate_left(18));
        h64 = xxh64_merge_round(h64, v1);
        h64 = xxh64_merge_round(h64, v2);
        h64 = xxh64_merge_round(h64, v3);
        h64 = xxh64_merge_round(h64, v4);
    } else {
        h64 = seed.wrapping_add(XXH64_PRIME5);
    }

    h64 = h64.wrapping_add(len as u64);

    while p + 8 <= len {
        let k1 = xxh64_round(0, read_u64_le(input, p));
        p += 8;
        h64 ^= k1;
        h64 = h64
            .rotate_left(27)
            .wrapping_mul(XXH64_PRIME1)
            .wrapping_add(XXH64_PRIME4);
    }

    if p + 4 <= len {
        h64 ^= read_u32_le(input, p).wrapping_mul(XXH64_PRIME1);
        p += 4;
        h64 = h64
            .rotate_left(23)
            .wrapping_mul(XXH64_PRIME2)
            .wrapping_add(XXH64_PRIME3);
    }

    while p < len {
        h64 ^= (input[p] as u64).wrapping_mul(XXH64_PRIME5);
        p += 1;
        h64 = h64.rotate_left(11).wrapping_mul(XXH64_PRIME1);
    }

    xxh64_avalanche(h64)
}

/// Builds the byte sequence the `cch` is hashed over: a copy of the body (carrying the `cch=00000`
/// placeholder) with the `model` value emptied and `max_tokens`, `fallbacks`, and
/// `fallback_credit_token` removed along with their separating comma.
fn filtered_preimage(body: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(body.len());
    let mut i = 0;

    while i < body.len() {
        if let Some((next, replacement, trim_prev_comma)) = filter_edit(body, i) {
            if trim_prev_comma && out.last() == Some(&b',') {
                out.pop();
            }
            out.extend_from_slice(replacement);
            i = next;
        } else {
            out.push(body[i]);
            i += 1;
        }
    }

    out
}

/// If a field to strip/normalize starts at `i`, returns `(next_index, replacement_bytes,
/// trim_preceding_comma)`. `trim_preceding_comma` removes a now-dangling comma left before a
/// deleted field that had no trailing comma of its own.
fn filter_edit(body: &[u8], i: usize) -> Option<(usize, &'static [u8], bool)> {
    const MODEL: &[u8] = b"\"model\":\"";
    const MODEL_EMPTY: &[u8] = b"\"model\":\"\"";
    const MAX_TOKENS: &[u8] = b"\"max_tokens\":";
    const FALLBACKS: &[u8] = b"\"fallbacks\":[";
    const FALLBACK_TOKEN: &[u8] = b"\"fallback_credit_token\":\"";

    if body[i..].starts_with(MODEL) {
        let end = json_string_end(body, i + MODEL.len())?;
        return Some((end + 1, MODEL_EMPTY, false));
    }

    if body[i..].starts_with(MAX_TOKENS) {
        let start = i + MAX_TOKENS.len();
        let end = digits_end(body, start);
        return (end > start).then(|| skip_field(body, i, end));
    }

    if body[i..].starts_with(FALLBACKS) {
        let array_start = i + FALLBACKS.len() - 1;
        let end = json_array_end(body, array_start)?;
        return Some(skip_field(body, i, end + 1));
    }

    if body[i..].starts_with(FALLBACK_TOKEN) {
        let end = json_string_end(body, i + FALLBACK_TOKEN.len())?;
        return Some(skip_field(body, i, end + 1));
    }

    None
}

/// Computes the edit that deletes the field spanning `start..end`, consuming a trailing comma if
/// present, otherwise signalling that a preceding comma must be trimmed.
fn skip_field(body: &[u8], start: usize, end: usize) -> (usize, &'static [u8], bool) {
    if body.get(end) == Some(&b',') {
        (end + 1, b"", false)
    } else {
        (end, b"", start > 0 && body[start - 1] == b',')
    }
}

/// Index of the closing quote of a JSON string whose contents start at `i` (handles `\\` escapes).
fn json_string_end(body: &[u8], mut i: usize) -> Option<usize> {
    while i < body.len() {
        match body[i] {
            b'\\' => i += 2,
            b'"' => return Some(i),
            _ => i += 1,
        }
    }
    None
}

/// Index of the closing bracket of a JSON array whose opening `[` is at `i`.
fn json_array_end(body: &[u8], mut i: usize) -> Option<usize> {
    let mut depth = 0usize;

    while i < body.len() {
        match body[i] {
            b'"' => i = json_string_end(body, i + 1)? + 1,
            b'[' => {
                depth += 1;
                i += 1;
            }
            b']' => {
                depth = depth.checked_sub(1)?;
                if depth == 0 {
                    return Some(i);
                }
                i += 1;
            }
            _ => i += 1,
        }
    }

    None
}

/// Index just past a run of ASCII digits starting at `i`.
fn digits_end(body: &[u8], mut i: usize) -> usize {
    while body.get(i).is_some_and(u8::is_ascii_digit) {
        i += 1;
    }
    i
}

/// Byte index just past the colon of the request body's *top-level* `"system"` key.
///
/// The scan is structural rather than a substring search: it tracks string boundaries (with
/// backslash escapes) and brace/bracket depth, and only considers a quoted token a key when it sits
/// at the root object's own level and is followed by a colon. Anything inside a message is
/// therefore invisible to it.
///
/// That is the whole point. The body puts `messages` ahead of `system`, matching Claude Code's key
/// order, so a conversation that quotes a billing header -- which is exactly what a session about
/// this code does -- would otherwise let a message win the search and take the attestation with it.
fn top_level_system_value(body: &[u8]) -> Option<usize> {
    let mut i = body.iter().position(|byte| !byte.is_ascii_whitespace())?;
    if body.get(i) != Some(&b'{') {
        return None;
    }
    i += 1;

    let mut depth = 0usize;
    while i < body.len() {
        match body[i] {
            b'"' => {
                let end = json_string_end(body, i + 1)?;
                if depth == 0 {
                    let mut after = end + 1;
                    while body.get(after).is_some_and(u8::is_ascii_whitespace) {
                        after += 1;
                    }
                    if body.get(after) == Some(&b':') && &body[i + 1..end] == b"system" {
                        return Some(after + 1);
                    }
                }
                i = end + 1;
            }
            b'{' | b'[' => {
                depth += 1;
                i += 1;
            }
            b'}' | b']' => {
                // Depth 0 here is the root object's own closing brace: no top-level `system`.
                depth = depth.checked_sub(1)?;
                i += 1;
            }
            _ => i += 1,
        }
    }

    None
}

/// Replaces the `cch=00000` placeholder with the attestation token. The search starts at the
/// top-level `system` array so no message can supply the match; the hash is taken over
/// [`filtered_preimage`] of the body.
pub(super) fn patch_request_body(body_json: &str) -> Result<String> {
    const BILLING_PREFIX: &str = "x-anthropic-billing-header:";
    const PLACEHOLDER: &str = "cch=00000";

    let system_start = top_level_system_value(body_json.as_bytes())
        .ok_or_else(|| MekaError::Provider("no top-level system array in request body".into()))?;

    let billing_start = body_json[system_start..]
        .find(BILLING_PREFIX)
        .map(|relative| system_start + relative)
        .ok_or_else(|| {
            MekaError::Provider("x-anthropic-billing-header not found in request body".into())
        })?;

    let idx = body_json[billing_start..]
        .find(PLACEHOLDER)
        .map(|relative| billing_start + relative)
        .ok_or_else(|| {
            MekaError::Provider(
                "cch=00000 attestation placeholder not found in billing header".into(),
            )
        })?;

    let preimage = filtered_preimage(body_json.as_bytes());
    let digest = xxh64(&preimage, CCH_XXH64_SEED);
    let token = format!("{:05x}", digest & 0xfffff);

    let mut patched = String::with_capacity(body_json.len());
    patched.push_str(&body_json[..idx + 4]); // up to and including "cch="
    patched.push_str(&token);
    patched.push_str(&body_json[idx + 9..]); // skip past "00000"
    Ok(patched)
}

/// Builds the User-Agent string matching claude-code's format.
fn claude_user_agent() -> String {
    format!("claude-cli/{} (external, cli)", CC_VERSION)
}

/// Stainless SDK / runtime versions. Must match the release corresponding to `CC_VERSION`. Values
/// verified against wire captures of real Claude Code traffic. The runtime reports as `node`
/// (Bun's Node.js compat layer) with a fixed version string.
const STAINLESS_RUNTIME: &str = "node";
const STAINLESS_RUNTIME_VERSION: &str = "v26.3.0";
const STAINLESS_SDK_VERSION: &str = "0.112.1";

/// Maps `std::env::consts::ARCH` to Node.js/Bun `process.arch` names.
fn stainless_arch() -> &'static str {
    match std::env::consts::ARCH {
        "x86_64" => "x64",
        "x86" => "ia32",
        "aarch64" => "arm64",
        "arm" => "arm",
        "s390x" => "s390x",
        "powerpc64" => "ppc64",
        other => other,
    }
}

fn stainless_os() -> &'static str {
    match std::env::consts::OS {
        "macos" => "MacOS",
        "windows" => "Windows",
        "linux" => "Linux",
        "freebsd" => "FreeBSD",
        other => other,
    }
}

/// Applies all HTTP headers Claude Code sends, in the order it sends them.
///
/// The order is not cosmetic: HTTP/2 preserves it, so it is as much a client signature as the
/// values are. What the 2.1.241 wire capture shows is the Stainless SDK's `Headers` object
/// serialised in a case-sensitive sort (uppercase before lowercase), then the transport's own
/// `Connection` / `Host` / `Accept-Encoding` / `Content-Length` after it. `reqwest`'s `HeaderMap`
/// iterates in insertion order, so inserting in that order reproduces it.
///
/// Two parts of it are outside meka's reach and stay different. Header *names* go out lowercased
/// (`http::HeaderName` normalises, and HTTP/2 requires it anyway, so this is invisible on the real
/// wire), and `reqwest` places `Accept-Encoding` first because its decompression layer installs it
/// before any per-request header.
pub(super) fn apply_headers(
    request: reqwest::RequestBuilder,
    auth_header_name: &str,
    auth_header_value: &str,
    session_id: &str,
    betas: Option<&str>,
) -> reqwest::RequestBuilder {
    let mut request = request
        .header("Accept", "application/json")
        // From the SDK's `authHeaders()`.
        .header(auth_header_name, auth_header_value)
        // From the SDK's `bodyHeaders()`.
        .header("Content-Type", "application/json")
        .header("User-Agent", claude_user_agent())
        // From Claude Code's `defaultHeaders()`.
        .header("X-Claude-Code-Session-Id", session_id)
        .header("X-Stainless-Arch", stainless_arch())
        .header("X-Stainless-Lang", "js")
        .header("X-Stainless-OS", stainless_os())
        .header("X-Stainless-Package-Version", STAINLESS_SDK_VERSION)
        .header("X-Stainless-Retry-Count", "0")
        .header("X-Stainless-Runtime", STAINLESS_RUNTIME)
        .header("X-Stainless-Runtime-Version", STAINLESS_RUNTIME_VERSION)
        .header("X-Stainless-Timeout", "600");

    if let Some(betas) = betas {
        request = request.header("anthropic-beta", betas);
    }

    request
        .header("anthropic-dangerous-direct-browser-access", "true")
        .header("anthropic-version", "2023-06-01")
        .header("x-app", "cli")
        // Per-request, not from an SDK helper.
        .header("x-client-request-id", Uuid::new_v4().to_string())
        .header("Connection", "keep-alive")
        .header("Accept-Encoding", "gzip, deflate, br, zstd")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_compute_fingerprint_from_messages_matches_manual() {
        let messages = vec![Message::user("hello world, this is a test message!")];
        let from_messages = compute_fingerprint_from_messages(&messages);
        let first_text = extract_first_user_message_text(&messages);
        let manual = compute_fingerprint(&first_text, CC_VERSION);
        assert_eq!(from_messages, manual);
    }

    #[test]
    fn test_fingerprint_known_values() {
        let fingerprint = compute_fingerprint("hello", CC_VERSION);
        assert_eq!(fingerprint.len(), 3);
        assert!(fingerprint.chars().all(|c| c.is_ascii_hexdigit()));

        let fingerprint2 = compute_fingerprint("hello", CC_VERSION);
        assert_eq!(fingerprint, fingerprint2);

        let fingerprint3 = compute_fingerprint("this is a longer test message!!", CC_VERSION);
        assert_ne!(fingerprint, fingerprint3);
    }

    #[test]
    fn test_fingerprint_empty_message() {
        let fingerprint = compute_fingerprint("", CC_VERSION);
        assert_eq!(fingerprint.len(), 3);
        assert!(fingerprint.chars().all(|c| c.is_ascii_hexdigit()));
    }

    #[test]
    fn test_extract_first_user_message_text() {
        let messages = vec![
            Message {
                role: Role::Assistant,
                content: vec![ContentBlock::Text {
                    text: "assistant text".to_string(),
                }],
            },
            Message::user("user text"),
        ];
        assert_eq!(extract_first_user_message_text(&messages), "user text");

        let empty: Vec<Message> = vec![];
        assert_eq!(extract_first_user_message_text(&empty), "");
    }

    #[test]
    fn test_fingerprint_boundary_length_messages() {
        let fp5 = compute_fingerprint("abcde", CC_VERSION);
        assert_eq!(fp5.len(), 3);

        let fp8 = compute_fingerprint("abcdefgh", CC_VERSION);
        assert_eq!(fp8.len(), 3);

        let fp21 = compute_fingerprint("abcdefghijklmnopqrstu", CC_VERSION);
        assert_eq!(fp21.len(), 3);

        assert_ne!(fp5, fp8);
        assert_ne!(fp8, fp21);
    }

    #[test]
    fn test_fingerprint_short_message_all_fallback() {
        let fp_short = compute_fingerprint("abc", CC_VERSION);
        let fp_empty = compute_fingerprint("", CC_VERSION);
        assert_eq!(fp_short, fp_empty);
    }

    #[test]
    fn test_fingerprint_multibyte_chars() {
        let msg = "日本語のテスト文字列を使ったメッセージです！！！";
        assert!(msg.chars().count() > 20);
        let fp = compute_fingerprint(msg, CC_VERSION);
        assert_eq!(fp.len(), 3);
        assert!(fp.chars().all(|c| c.is_ascii_hexdigit()));

        assert_eq!(msg.chars().nth(4), Some('テ'));
        assert_eq!(msg.chars().nth(7), Some('文'));
        assert_eq!(msg.chars().nth(20), Some('す'));
    }

    #[test]
    fn test_fingerprint_different_version() {
        let fp_a = compute_fingerprint("hello", "1.0.0");
        let fp_b = compute_fingerprint("hello", "2.0.0");
        assert_eq!(fp_a.len(), 3);
        assert_eq!(fp_b.len(), 3);
        assert_ne!(fp_a, fp_b);
    }

    #[test]
    fn test_extract_first_user_message_text_no_text_block() {
        use crate::provider::ToolResultContent;
        let messages = vec![Message {
            role: Role::User,
            content: vec![ContentBlock::ToolResult {
                tool_use_id: "toolu_1".to_string(),
                content: vec![ToolResultContent::Text {
                    text: "result".to_string(),
                }],
                is_error: false,
            }],
        }];
        assert_eq!(extract_first_user_message_text(&messages), "");
    }

    #[test]
    fn test_extract_first_user_message_text_multiple_users() {
        let messages = vec![
            Message::user("first user message"),
            Message::user("second user message"),
        ];
        assert_eq!(
            extract_first_user_message_text(&messages),
            "first user message"
        );
    }

    #[test]
    fn test_extract_first_user_message_text_only_assistants() {
        let messages = vec![
            Message::assistant_text("hello"),
            Message::assistant_text("world"),
        ];
        assert_eq!(extract_first_user_message_text(&messages), "");
    }

    #[test]
    fn test_compute_fingerprint_from_messages_empty() {
        let empty: Vec<Message> = vec![];
        assert_eq!(
            compute_fingerprint_from_messages(&empty),
            compute_fingerprint("", CC_VERSION)
        );
    }

    #[test]
    fn test_compute_fingerprint_from_messages_no_user() {
        let messages = vec![Message::assistant_text("I'm an assistant")];
        assert_eq!(
            compute_fingerprint_from_messages(&messages),
            compute_fingerprint("", CC_VERSION)
        );
    }

    // All xxHash64 expected values cross-validated against Python xxhash.

    #[test]
    fn test_xxh64_basic() {
        assert_eq!(xxh64(b"", 0), 0xef46db3751d8e999);
        assert_eq!(xxh64(b"abc", 0), 0x44bc2cf5ad770999);
    }

    #[test]
    fn test_xxh64_claude_seed_short_body() {
        // No volatile fields, so the preimage equals the raw body; pins the seed.
        let body = r#"{"test":"cch=00000"}"#;
        let digest = xxh64(body.as_bytes(), CCH_XXH64_SEED);
        let token = format!("{:05x}", digest & 0xfffff);
        assert_eq!(token, "2f60d");
    }

    #[test]
    fn test_filtered_preimage_strips_volatile_fields() {
        // model emptied; max_tokens, fallbacks, fallback_credit_token removed with their commas.
        let body = br#"{"model":"claude-x","max_tokens":1024,"a":1,"fallbacks":[{"x":1}],"fallback_credit_token":"tok","b":2}"#;
        let preimage = String::from_utf8(filtered_preimage(body)).unwrap();
        assert_eq!(preimage, r#"{"model":"","a":1,"b":2}"#);
    }

    #[test]
    fn test_patch_request_body_realistic() {
        let body = concat!(
            r#"{"system":[{"type":"text","text":"x-anthropic-billing-header: "#,
            r#"cc_version=2.1.185.abc; cc_entrypoint=cli; cch=00000;"}],"model""#,
            r#":"claude-opus-4-20250514","max_tokens":1024,"messages":[{"role""#,
            r#":"user","content":"hi"}]}"#,
        );

        let patched = patch_request_body(body).unwrap();
        assert!(patched.contains("cch=16a13;"), "got: {patched}");
        assert!(!patched.contains("cch=00000"));
    }

    #[test]
    fn test_xxh64_one_byte() {
        assert_eq!(xxh64(b"x", 0), 0x5c80c09683041123);
    }

    #[test]
    fn test_xxh64_three_bytes() {
        assert_eq!(xxh64(b"abc", 0), 0x44bc2cf5ad770999);
    }

    #[test]
    fn test_xxh64_four_bytes() {
        assert_eq!(xxh64(b"abcd", 0), 0xde0327b0d25d92cc);
    }

    #[test]
    fn test_xxh64_seven_bytes() {
        assert_eq!(xxh64(b"abcdefg", 0), 0x1860940e2902822d);
    }

    #[test]
    fn test_xxh64_eight_bytes() {
        assert_eq!(xxh64(b"abcdefgh", 0), 0x3ad351775b4634b7);
    }

    #[test]
    fn test_xxh64_sixteen_bytes() {
        assert_eq!(xxh64(b"abcdefghijklmnop", 0), 0x71ce8137ca2dd53d);
    }

    #[test]
    fn test_xxh64_thirty_one_bytes() {
        let input = b"abcdefghijklmnopqrstuvwxyz01234";
        assert_eq!(input.len(), 31);
        assert_eq!(xxh64(input, 0), 0x16058c7b947da137);
    }

    #[test]
    fn test_xxh64_thirty_two_bytes() {
        let input = b"abcdefghijklmnopqrstuvwxyz012345";
        assert_eq!(input.len(), 32);
        assert_eq!(xxh64(input, 0), 0xbf2cd639b4143b80);
    }

    #[test]
    fn test_xxh64_with_nonzero_seed() {
        let input = b"hello world";
        let h0 = xxh64(input, 0);
        let h1 = xxh64(input, 1);
        let h_claude = xxh64(input, CCH_XXH64_SEED);
        assert_ne!(h0, h1);
        assert_ne!(h0, h_claude);
        assert_ne!(h1, h_claude);
    }

    #[test]
    fn test_patch_request_body_missing_billing_header() {
        let body = r#"{"system":[],"messages":[]}"#;
        let result = patch_request_body(body);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("x-anthropic-billing-header not found"));
    }

    #[test]
    fn test_patch_request_body_billing_header_without_placeholder() {
        let body = r#"{"system":[{"type":"text","text":"x-anthropic-billing-header: cc_version=2.1.86.abc; cc_entrypoint=cli;"}]}"#;
        let result = patch_request_body(body);
        assert!(result.is_err());
        let err_msg = result.unwrap_err().to_string();
        assert!(err_msg.contains("cch=00000"));
    }

    #[test]
    fn test_claude_user_agent_format() {
        let ua = claude_user_agent();
        assert!(ua.starts_with("claude-cli/"));
        assert!(ua.contains(CC_VERSION));
        assert!(ua.ends_with("(external, cli)"));
    }

    #[test]
    fn test_generate_billing_header_format() {
        let messages = vec![Message::user("hello")];
        let header = generate_billing_header(&messages);
        assert!(header.starts_with("x-anthropic-billing-header:"));
        assert!(header.contains(&format!("cc_version={}", CC_VERSION)));
        assert!(header.contains("cc_entrypoint=cli"));
        assert!(header.contains("cch=00000"));
        assert!(header.ends_with("cch=00000;"));

        // Fingerprint suffix is dynamic per first user message: different first message →
        // different suffix.
        let other =
            generate_billing_header(&[Message::user("totally different first user message text")]);
        assert_ne!(header, other);
    }

    #[test]
    fn test_stainless_arch_returns_nonempty() {
        assert!(!stainless_arch().is_empty());
    }
}
