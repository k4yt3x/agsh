//! Terminal rendering: streaming markdown renderer (syntect highlighting + termimad), tool-call
//! indicators, todo-list display, and helpers for one-off CLI status/error messages. Owns the
//! embedded Monokai Extended theme used for code blocks.

use std::{
    io::{self, Write},
    sync::{LazyLock, OnceLock},
};

use crossterm::style::{Color, Stylize};
use regex::Regex;
use syntect::{
    easy::HighlightLines,
    highlighting::{Theme, ThemeSet},
    parsing::{SyntaxReference, SyntaxSet},
    util::{LinesWithEndings, as_24_bit_terminal_escaped},
};
use termimad::MadSkin;

/// Monokai Extended theme, vendored from bat's `sharkdp/sublime-monokai-extended` (MIT).
const MONOKAI_EXTENDED_TMTHEME: &[u8] = include_bytes!("../assets/themes/Monokai Extended.tmTheme");

enum LastOutput {
    Nothing,
    Prompt,
    Text,
    Thinking,
    ToolIndicator,
    TodoList,
}

/// Tracks what was last printed to decide if a blank line is needed next.
pub struct OutputSpacing {
    last: LastOutput,
}

impl OutputSpacing {
    pub fn new() -> Self {
        Self {
            last: LastOutput::Nothing,
        }
    }

    /// Call before printing streamed text. Returns true if a blank line should be emitted first.
    pub fn before_text(&mut self) -> bool {
        let need_blank = matches!(self.last, LastOutput::ToolIndicator | LastOutput::Thinking);
        self.last = LastOutput::Text;
        need_blank
    }

    /// Call before printing a tool indicator. Returns true if a blank line should be emitted first.
    pub fn before_tool_indicator(&mut self) -> bool {
        let need_blank = matches!(self.last, LastOutput::Text | LastOutput::Thinking);
        self.last = LastOutput::ToolIndicator;
        need_blank
    }

    /// Call before printing a thinking block. Returns true if a blank line should be emitted first.
    pub fn before_thinking(&mut self) -> bool {
        let need_blank = matches!(self.last, LastOutput::Text | LastOutput::ToolIndicator);
        self.last = LastOutput::Thinking;
        need_blank
    }

    /// Call after the todo list is rendered (it has its own trailing newline).
    pub fn after_todo_list(&mut self) {
        self.last = LastOutput::TodoList;
    }

    /// Call after newline_after_prompt is printed.
    pub fn after_prompt(&mut self) {
        self.last = LastOutput::Prompt;
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, serde::Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum RenderMode {
    /// syntect-based highlighter (default). Named after the `syntect` crate that does the
    /// in-process highlighting.
    #[default]
    Syntect,
    Termimad,
    Raw,
    /// Emits no output to stdout/stderr. Used by sub-agents and any other in-process
    /// [`crate::agent::Agent`] that shouldn't leak to the user's terminal. The
    /// [`StreamingRenderer`] no-ops for this mode and the `render::*` helpers short-circuit on
    /// `matches!(mode, RenderMode::Silent)` at each call site.
    Silent,
}

impl std::fmt::Display for RenderMode {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            RenderMode::Syntect => write!(formatter, "syntect"),
            RenderMode::Termimad => write!(formatter, "termimad"),
            RenderMode::Raw => write!(formatter, "raw"),
            RenderMode::Silent => write!(formatter, "silent"),
        }
    }
}

impl std::str::FromStr for RenderMode {
    type Err = String;

    fn from_str(string: &str) -> std::result::Result<Self, Self::Err> {
        match string.to_lowercase().as_str() {
            "syntect" => Ok(RenderMode::Syntect),
            "rich" | "termimad" => Ok(RenderMode::Termimad),
            "raw" => Ok(RenderMode::Raw),
            "silent" => Ok(RenderMode::Silent),
            other => Err(format!(
                "unknown render mode '{}' (expected 'syntect', 'termimad', 'raw', or 'silent')",
                other
            )),
        }
    }
}

pub struct StreamingRenderer {
    buffer: String,
    skin: MadSkin,
    mode: RenderMode,
    pub(crate) started: bool,
    raw_table_lines: Vec<String>,
    code_block_lines: Vec<String>,
}

impl StreamingRenderer {
    pub fn new(mode: RenderMode) -> Self {
        Self {
            buffer: String::new(),
            skin: MadSkin::default_dark(),
            mode,
            started: false,
            raw_table_lines: Vec::new(),
            code_block_lines: Vec::new(),
        }
    }

    pub fn push_delta(&mut self, delta: &str) -> io::Result<()> {
        // Short-circuit before any buffering; Silent shouldn't even accumulate state since
        // `finish` will discard it anyway.
        if matches!(self.mode, RenderMode::Silent) {
            return Ok(());
        }

        let delta = if self.started {
            delta
        } else {
            let trimmed = delta.trim_start_matches('\n');
            if trimmed.is_empty() {
                return Ok(());
            }
            self.started = true;
            trimmed
        };

        self.buffer.push_str(delta);

        match self.mode {
            RenderMode::Syntect => self.flush_syntect(),
            RenderMode::Termimad => self.flush_termimad(),
            RenderMode::Raw => self.flush_raw(),
            RenderMode::Silent => Ok(()),
        }
    }

    pub fn finish(&mut self) -> io::Result<()> {
        if matches!(self.mode, RenderMode::Silent) {
            return Ok(());
        }
        match self.mode {
            RenderMode::Syntect => {
                if !self.buffer.is_empty() {
                    let remaining = std::mem::take(&mut self.buffer);
                    let trimmed = remaining.trim_end_matches('\n');
                    let mut needs_newline = false;
                    for line in trimmed.lines() {
                        let is_fence = line.trim_start().starts_with("```");

                        if !self.code_block_lines.is_empty() {
                            self.code_block_lines.push(line.to_string());
                            if is_fence {
                                self.flush_syntect_code_block()?;
                                needs_newline = false;
                            }
                        } else if is_fence {
                            self.flush_syntect_table()?;
                            self.code_block_lines.push(line.to_string());
                            needs_newline = false;
                        } else if is_table_line(line) {
                            self.raw_table_lines.push(line.to_string());
                            needs_newline = false;
                        } else if line.is_empty() {
                            self.flush_syntect_table()?;
                            println!();
                            needs_newline = false;
                        } else {
                            self.flush_syntect_table()?;
                            print_highlighted_markdown(line);
                            needs_newline = true;
                        }
                    }
                    self.flush_syntect_code_block()?;
                    self.flush_syntect_table()?;
                    if needs_newline {
                        println!();
                    }
                }
            }
            RenderMode::Termimad => {
                if !self.buffer.is_empty() {
                    let remaining = std::mem::take(&mut self.buffer);
                    let trimmed = remaining.trim_end_matches('\n');
                    if !trimmed.is_empty() {
                        print!("{}", self.skin.term_text(trimmed));
                    }
                }
            }
            RenderMode::Raw => {
                if !self.buffer.is_empty() {
                    let remaining = std::mem::take(&mut self.buffer);
                    let trimmed = remaining.trim_end_matches('\n');
                    for line in trimmed.lines() {
                        if is_table_line(line) {
                            self.raw_table_lines.push(line.to_string());
                        } else {
                            self.flush_raw_table()?;
                            println!("{}", line);
                        }
                    }
                    self.flush_raw_table()?;
                }
            }
            // Already short-circuited above; included for exhaustiveness.
            RenderMode::Silent => {}
        }
        io::stdout().flush()
    }

    fn flush_syntect(&mut self) -> io::Result<()> {
        self.buffer = normalize_spacing(&self.buffer);

        while let Some(newline_pos) = self.buffer.find('\n') {
            let line = self.buffer[..newline_pos].to_string();
            let is_fence = line.trim_start().starts_with("```");

            // If we're inside a code block, accumulate lines
            if !self.code_block_lines.is_empty() {
                self.buffer = self.buffer[newline_pos + 1..].to_string();
                self.code_block_lines.push(line);
                if is_fence {
                    self.flush_syntect_code_block()?;
                }
                continue;
            }

            // Opening fence starts a new code block
            if is_fence {
                self.buffer = self.buffer[newline_pos + 1..].to_string();
                self.flush_syntect_table()?;
                self.code_block_lines.push(line);
                continue;
            }

            self.buffer = self.buffer[newline_pos + 1..].to_string();

            if is_table_line(&line) {
                self.raw_table_lines.push(line);
            } else {
                self.flush_syntect_table()?;
                if line.is_empty() {
                    println!();
                } else {
                    print_highlighted_markdown(&format!("{}\n", line));
                }
                io::stdout().flush()?;
            }
        }
        Ok(())
    }

    fn flush_syntect_code_block(&mut self) -> io::Result<()> {
        if self.code_block_lines.is_empty() {
            return Ok(());
        }
        let lines = std::mem::take(&mut self.code_block_lines);
        print!("{}", render_code_block_to_string(&lines));
        io::stdout().flush()
    }

    fn flush_syntect_table(&mut self) -> io::Result<()> {
        if self.raw_table_lines.is_empty() {
            return Ok(());
        }

        let lines = std::mem::take(&mut self.raw_table_lines);
        let formatted = format_table(&lines);
        let table_text = formatted.join("\n");
        print_highlighted_markdown(&table_text);
        println!();
        io::stdout().flush()
    }

    fn flush_termimad(&mut self) -> io::Result<()> {
        self.buffer = normalize_spacing(&self.buffer);

        while let Some(boundary) = self.buffer.find("\n\n") {
            let complete = self.buffer[..boundary + 2].to_string();
            self.buffer = self.buffer[boundary + 2..].to_string();
            print!("{}", self.skin.term_text(&complete));
            io::stdout().flush()?;
        }

        if !self.in_code_block() && !self.in_table() {
            while let Some(newline_pos) = self.buffer.find('\n') {
                if newline_pos + 1 < self.buffer.len() || !self.buffer.ends_with('\n') {
                    let line = self.buffer[..newline_pos + 1].to_string();
                    self.buffer = self.buffer[newline_pos + 1..].to_string();
                    print!("{}", self.skin.term_text(&line));
                    io::stdout().flush()?;
                } else {
                    break;
                }
            }
        }

        Ok(())
    }

    fn flush_raw(&mut self) -> io::Result<()> {
        self.buffer = normalize_spacing(&self.buffer);

        while let Some(newline_pos) = self.buffer.find('\n') {
            let line = self.buffer[..newline_pos].to_string();
            self.buffer = self.buffer[newline_pos + 1..].to_string();

            if is_table_line(&line) {
                self.raw_table_lines.push(line);
            } else {
                self.flush_raw_table()?;
                println!("{}", line);
                io::stdout().flush()?;
            }
        }
        Ok(())
    }

    fn flush_raw_table(&mut self) -> io::Result<()> {
        if self.raw_table_lines.is_empty() {
            return Ok(());
        }

        let lines = std::mem::take(&mut self.raw_table_lines);
        let formatted = format_table(&lines);
        for line in &formatted {
            println!("{}", line);
        }
        io::stdout().flush()
    }

    fn in_code_block(&self) -> bool {
        let fence_count = self.buffer.matches("```").count();
        !fence_count.is_multiple_of(2)
    }

    fn in_table(&self) -> bool {
        self.buffer.trim_start().starts_with('|')
    }
}

/// Ensure blank lines after markdown headers and tables when followed by non-empty content. Skips
/// content inside code fences.
fn normalize_spacing(text: &str) -> String {
    let lines: Vec<&str> = text.lines().collect();
    let mut result = Vec::with_capacity(lines.len());
    let mut in_fence = false;

    for (index, line) in lines.iter().enumerate() {
        if line.trim_start().starts_with("```") {
            in_fence = !in_fence;
        }

        result.push(*line);

        if in_fence {
            continue;
        }

        let next_line = lines.get(index + 1);
        let next_is_non_empty = next_line.is_some_and(|next| !next.trim().is_empty());

        if !next_is_non_empty {
            continue;
        }

        let trimmed = line.trim_start();

        // Blank line after headers (e.g., `## Title`)
        let is_header = trimmed.starts_with('#')
            && trimmed
                .find(|character: char| character != '#')
                .is_some_and(|position| trimmed.as_bytes().get(position) == Some(&b' '));

        // Blank line after table rows when next line is clearly not a table row. A line starting
        // with `|` might be an incomplete table row from streaming, so only treat lines NOT
        // starting with `|` as table-ending.
        let is_table_end = is_table_line(line)
            && next_line.is_some_and(|next| !next.trim_start().starts_with('|'));

        if is_header || is_table_end {
            result.push("");
        }
    }

    // Preserve trailing newline if the original had one
    let mut output = result.join("\n");
    if text.ends_with('\n') {
        output.push('\n');
    }
    output
}

/// Holds the expensive-to-load syntect assets, a `SyntaxSet` (~1 MB bincode blob) and a dark
/// `Theme`, so subsequent highlighting calls can reuse them without paying the decode cost each
/// time. Session-resume reprint and live streaming both call `highlight_markdown_line` per line;
/// initializing assets once per process turns that cost from ~50 ms/call into <1 ms/call.
struct Highlighter {
    syntax_set: SyntaxSet,
    theme: Theme,
}

static HIGHLIGHTER: OnceLock<Highlighter> = OnceLock::new();

// The `expect()` below loads a compile-time `include_bytes!()` of the bundled theme; a parse
// failure would mean we shipped a corrupt `.tmTheme` resource, caught on the first build/test.
#[allow(clippy::expect_used)]
fn highlighter() -> &'static Highlighter {
    HIGHLIGHTER.get_or_init(|| {
        let syntax_set = SyntaxSet::load_defaults_newlines();
        let mut cursor = std::io::Cursor::new(MONOKAI_EXTENDED_TMTHEME);
        let theme =
            ThemeSet::load_from_reader(&mut cursor).expect("embedded Monokai Extended theme loads");
        Highlighter { syntax_set, theme }
    })
}

/// Syntax-highlight a chunk of markdown and write it to stdout with 24-bit ANSI color escapes. The
/// caller is responsible for any surrounding newlines.
fn print_highlighted_markdown(text: &str) {
    let output = highlight_markdown_to_string(text);
    print!("{}", output);
}

/// Returns the ANSI-escaped highlighted text without writing to stdout. Exposed for testing.
fn highlight_markdown_to_string(text: &str) -> String {
    let highlighter = highlighter();
    let syntax = highlighter
        .syntax_set
        .find_syntax_by_name("Markdown")
        .or_else(|| highlighter.syntax_set.find_syntax_by_extension("md"))
        .unwrap_or_else(|| highlighter.syntax_set.find_syntax_plain_text());
    highlight_with_syntax(text, syntax)
}

/// Highlight `text` line-by-line with an explicit syntect grammar and return 24-bit ANSI escapes.
/// Shared core of the Markdown-prose path and the per-language code-block path; the passed `syntax`
/// must come from the same static [`highlighter`] so its context indices match the `syntax_set`
/// used to resolve embeds.
fn highlight_with_syntax(text: &str, syntax: &SyntaxReference) -> String {
    let highlighter = highlighter();
    let mut highlight = HighlightLines::new(syntax, &highlighter.theme);

    let mut out = String::new();
    for line in LinesWithEndings::from(text) {
        match highlight.highlight_line(line, &highlighter.syntax_set) {
            Ok(ranges) => {
                out.push_str(&as_24_bit_terminal_escaped(&ranges[..], false));
            }
            Err(error) => {
                // On parse error, fall back to plain text so we never lose content.
                tracing::debug!("syntect highlight failed: {}", error);
                out.push_str(line);
            }
        }
    }
    // Reset ANSI so colors don't bleed into the next prompt.
    out.push_str("\x1b[0m");
    out
}

/// A markdown code-fence line, e.g. ```` ```rust ```` or a bare ```` ``` ````. Indentation is
/// tolerated.
fn is_code_fence(line: &str) -> bool {
    line.trim_start().starts_with("```")
}

/// Extract the language token from an opening code fence: ```` ```rust ````→`Some("rust")`,
/// ```` ``` ````→`None`, ```` ```rust,ignore ````/```` ```js title=x ````→the first token. The
/// token is whatever precedes the first whitespace or comma after the backticks.
fn parse_fence_language(fence_line: &str) -> Option<&str> {
    let after = fence_line.trim_start().trim_start_matches('`');
    let token = after
        .split(|character: char| character.is_whitespace() || character == ',')
        .next()
        .unwrap_or("")
        .trim();
    (!token.is_empty()).then_some(token)
}

/// Resolve a fence language tag to a syntect grammar, falling back to plain text for an absent or
/// unrecognized tag (which then renders uncolored rather than erroring). `find_syntax_by_token` is
/// already case-insensitive on both extension and grammar name in syntect 5.x, so lowercase tags
/// like `rust`/`python`/`json` resolve directly; the alias arm only covers tags that are neither a
/// file extension nor a grammar name.
fn syntax_for_language(lang: Option<&str>) -> &'static SyntaxReference {
    let set = &highlighter().syntax_set;
    let Some(lang) = lang.map(str::trim).filter(|value| !value.is_empty()) else {
        return set.find_syntax_plain_text();
    };
    let resolved = match lang.to_ascii_lowercase().as_str() {
        "text" | "plain" | "plaintext" => Some(set.find_syntax_plain_text()),
        "shell" | "console" => set.find_syntax_by_extension("sh"),
        _ => set.find_syntax_by_token(lang),
    };
    resolved.unwrap_or_else(|| set.find_syntax_plain_text())
}

/// Render a fenced code block to ANSI-highlighted text. `lines` is
/// `[opening_fence, body…, closing_fence?]` (the closing fence is absent when the stream ended
/// mid-block). The fence lines keep the Markdown coloring they've always had; the body is
/// highlighted with the block's own language grammar (falling back to plain text for an
/// absent/unknown tag). Every line ends in a single `\n`, matching the prior `join("\n")` +
/// `println!()` output.
fn render_code_block_to_string(lines: &[String]) -> String {
    if lines.is_empty() {
        return String::new();
    }
    let language = parse_fence_language(&lines[0]);
    let has_closing = lines.len() > 1 && is_code_fence(&lines[lines.len() - 1]);
    let body_end = if has_closing {
        lines.len() - 1
    } else {
        lines.len()
    };

    let mut out = highlight_markdown_to_string(&format!("{}\n", lines[0]));
    if body_end > 1 {
        let body: String = lines[1..body_end]
            .iter()
            .map(|line| format!("{}\n", line))
            .collect();
        out.push_str(&highlight_with_syntax(&body, syntax_for_language(language)));
    }
    if has_closing {
        out.push_str(&highlight_markdown_to_string(&format!(
            "{}\n",
            lines[body_end]
        )));
    }
    out
}

fn is_table_line(line: &str) -> bool {
    let trimmed = line.trim();
    trimmed.starts_with('|') && trimmed.ends_with('|') && trimmed.len() > 1
}

fn is_separator_row(cells: &[String]) -> bool {
    cells.iter().all(|cell| {
        cell.chars()
            .all(|character| character == '-' || character == ':')
    })
}

fn parse_table_row(line: &str) -> Vec<String> {
    let trimmed = line.trim();
    let inner = trimmed
        .strip_prefix('|')
        .unwrap_or(trimmed)
        .strip_suffix('|')
        .unwrap_or(trimmed);
    inner
        .split('|')
        .map(|cell| cell.trim().to_string())
        .collect()
}

fn display_width(string: &str) -> usize {
    unicode_width::UnicodeWidthStr::width(string)
}

fn format_table(lines: &[String]) -> Vec<String> {
    if lines.is_empty() {
        return Vec::new();
    }

    let parsed: Vec<Vec<String>> = lines.iter().map(|line| parse_table_row(line)).collect();

    let column_count = parsed.iter().map(|row| row.len()).max().unwrap_or(0);
    if column_count == 0 {
        return lines.to_vec();
    }

    let mut column_widths = vec![0usize; column_count];
    for row in &parsed {
        if is_separator_row(row) {
            continue;
        }
        for (column_index, cell) in row.iter().enumerate() {
            if column_index < column_count {
                column_widths[column_index] = column_widths[column_index].max(display_width(cell));
            }
        }
    }

    // Ensure minimum width of 3 for separator dashes
    for width in &mut column_widths {
        *width = (*width).max(3);
    }

    let mut result = Vec::new();
    for row in &parsed {
        if is_separator_row(row) {
            let separator: Vec<String> = column_widths
                .iter()
                .map(|width| "-".repeat(*width))
                .collect();
            result.push(format!("| {} |", separator.join(" | ")));
        } else {
            let padded: Vec<String> = (0..column_count)
                .map(|column_index| {
                    let cell = row.get(column_index).map(|s| s.as_str()).unwrap_or("");
                    let padding = column_widths[column_index].saturating_sub(display_width(cell));
                    format!("{}{}", cell, " ".repeat(padding))
                })
                .collect();
            result.push(format!("| {} |", padded.join(" | ")));
        }
    }

    result
}

/// Render the live "[tool X(`arg`)]" indicator line on stderr. The agent loop computes
/// `display_summary` (via [`resolve_primary_param`] over the tool's JSON Schema) and passes it
/// pre-resolved so the frontend layer no longer needs the schema at all. See
/// `FrontendEvent::ToolCallStarted` in `crate::frontend`.
pub fn render_tool_indicator(
    name: &str,
    _input: &serde_json::Value,
    display_summary: Option<&str>,
) {
    let display_name = tool_display_name(name);
    let indicator = match display_summary {
        Some(value) => {
            // Strip ANSI escapes and C0 control chars before display so a model-supplied command or
            // path can't spoof the permission prompt, clear the screen, or move the cursor. The
            // LLM-facing copy keeps the raw bytes.
            let sanitized = sanitize_for_display(&value.replace('\n', " "));
            let truncated = truncate_display(&sanitized, 80);
            format!("[tool {}(`{}`)]", display_name, truncated)
        }
        None => format!("[tool {}]", display_name),
    };
    eprintln!("{}", indicator.with(Color::DarkCyan));
}

/// Match ANSI CSI (Control Sequence Introducer) escapes: `ESC [` followed by parameter bytes
/// (`0x30-0x3F`), optional intermediate bytes (`0x20-0x2F`), and a final byte (`0x40-0x7E`). This
/// covers the sequences an attacker would use to clear the screen, move the cursor, or alter
/// colors.
// Compile-time regex literal; a `Regex::new` failure here means we shipped a typo in the pattern,
// caught on first build.
#[allow(clippy::expect_used)]
static CSI_PATTERN: LazyLock<Regex> = LazyLock::new(|| {
    Regex::new(r"\x1b\[[\x30-\x3f]*[\x20-\x2f]*[\x40-\x7e]").expect("static CSI pattern")
});

/// Strip ANSI CSI escapes and C0 control characters (except `\n`, `\r`, `\t`) from a string
/// destined for the user's terminal. Intended for text that originates in untrusted sources (LLM
/// tool arguments, command output echoed into indicators/prompts, etc.) so a hostile or broken
/// string cannot forge UI chrome or corrupt terminal state.
///
/// The sanitized form is for **display only**. The conversation copy sent back to the LLM keeps
/// full fidelity.
pub fn sanitize_for_display(text: &str) -> String {
    let stripped = CSI_PATTERN.replace_all(text, "");
    stripped
        .chars()
        .filter(|c| !c.is_control() || matches!(c, '\n' | '\r' | '\t'))
        .collect()
}

pub fn render_session_id(label: &str, id: &str) {
    eprintln!("{}", format!("{}: {}", label, id).with(Color::DarkGrey));
}

/// Format `rows` into a left-aligned, space-padded column layout, the shared renderer for meka's
/// CLI list tables (`skill list`, `mcp list`, `list`, `scratchpad_list`).
///
/// Each column is widened to its longest cell, the matching header included. Columns are separated
/// by two spaces; the final column is left unpadded so a long trailing value (a path, a URL, a
/// preview) doesn't drag a run of trailing whitespace. The returned string has one trailing newline
/// per line and no extra blank line; the caller picks the stream (`print!` for stdout list
/// commands, or embed it in a tool result).
///
/// (Distinct from the private `format_table`, which lays out *markdown* pipe tables for the
/// streaming renderer.)
///
/// Width is measured in `char`s, which is correct for the ASCII-dominated data meka tabulates
/// (names, versions, UUIDs, timestamps); a CJK-heavy cell would pad slightly short. No caller hits
/// that today.
pub fn format_columns(headers: &[&str], rows: &[Vec<String>]) -> String {
    if headers.is_empty() {
        return String::new();
    }

    let mut widths: Vec<usize> = headers.iter().map(|h| h.chars().count()).collect();
    for row in rows {
        for (index, cell) in row.iter().take(widths.len()).enumerate() {
            widths[index] = widths[index].max(cell.chars().count());
        }
    }

    let mut out = format_columns_row(headers, &widths);
    for row in rows {
        let cells: Vec<&str> = row.iter().map(String::as_str).collect();
        out.push_str(&format_columns_row(&cells, &widths));
    }
    out
}

fn format_columns_row(cells: &[&str], widths: &[usize]) -> String {
    use std::fmt::Write as _;

    let mut line = String::new();
    let last = cells.len().saturating_sub(1);
    for (index, cell) in cells.iter().enumerate() {
        if index == last {
            // Final column: never padded; nothing follows it.
            line.push_str(cell);
        } else {
            let width = widths.get(index).copied().unwrap_or(0);
            let _ = write!(line, "{:<w$}  ", cell, w = width);
        }
    }
    line.push('\n');
    line
}

pub fn render_hint(message: &str) {
    eprintln!("{}", message.with(Color::DarkGrey));
}

pub(crate) fn format_token_count(n: u64) -> String {
    if n < 1_000 {
        n.to_string()
    } else if n < 1_000_000 {
        format!("{:.1}k", (n as f64) / 1_000.0)
    } else {
        format!("{:.1}M", (n as f64) / 1_000_000.0)
    }
}

/// Print a one-line per-turn token-usage summary to stderr in dark grey, preceded by a blank line
/// so it visually separates from the agent's response. Format: `[in 12.3k / cache hit 96% / out
/// 1.2k]`. The "in" column is the total of all three input-token tiers (live, cache-write,
/// cache-read); the cache-hit % is `cache_read / total_in`. Numbers below 1k show as raw counts,
/// below 1M as `Nk`, and otherwise as `NM`, each with one decimal.
pub fn render_token_usage(usage: &crate::provider::TokenUsage) {
    let total_in = usage
        .input_tokens
        .saturating_add(usage.cache_creation_input_tokens)
        .saturating_add(usage.cache_read_input_tokens);
    let cache_hit_pct = if total_in == 0 {
        0
    } else {
        ((usage.cache_read_input_tokens as f64) / (total_in as f64) * 100.0).round() as u64
    };
    eprintln!();
    eprintln!(
        "{}",
        format!(
            "[in {} / cache hit {}% / out {}]",
            format_token_count(total_in),
            cache_hit_pct,
            format_token_count(usage.output_tokens),
        )
        .with(Color::DarkGrey)
    );
}

/// The resolved model parameters shown at the top of the `/status` report, borrowed from the active
/// config plus the provider's settled effort. All optional so a mis-selected profile still renders.
pub struct ModelStatus<'a> {
    pub model: Option<&'a str>,
    /// Active profile name (e.g. `claude-max`).
    pub profile: Option<&'a str>,
    /// Backend type (e.g. `claude-oauth`).
    pub backend: Option<&'a str>,
    /// The reasoning effort sent on the wire, or `None` when the model sends none.
    pub effort: Option<&'a str>,
    pub thinking: bool,
}

/// Multi-line cumulative session report shown by the `/status` slash command. Goes to stderr
/// (matches the rest of REPL UI feedback).
pub fn render_session_status(
    snap: &crate::stats::SessionStatsSnapshot,
    model: &ModelStatus,
    message_count: usize,
    context_tokens: u64,
    context_window: u64,
) {
    let total_in = snap.total_input_tokens();
    let header = "Session status".with(Color::Cyan);
    eprintln!("{}", header);
    if let Some(name) = model.model {
        eprintln!("  Model:           {}", name);
    }
    match (model.profile, model.backend) {
        (Some(profile), Some(backend)) => eprintln!("  Provider:        {} ({})", profile, backend),
        (None, Some(backend)) => eprintln!("  Provider:        {}", backend),
        _ => {}
    }
    if let Some(effort) = model.effort {
        eprintln!("  Effort:          {}", effort);
    }
    eprintln!(
        "  Thinking:        {}",
        if model.thinking { "on" } else { "off" }
    );
    eprintln!("  Turns:           {}", snap.turns);
    // Live context occupancy: how full the window was on the last request. Distinct from the
    // cumulative "Input tokens" total below, which sums every turn's usage for the whole session.
    if context_window > 0 && context_tokens > 0 {
        let pct = ((context_tokens as f64 / context_window as f64) * 100.0).round() as u64;
        let remaining = context_window.saturating_sub(context_tokens);
        eprintln!(
            "  Context:         {} / {} ({}% used, {} left)",
            format_token_count(context_tokens),
            format_token_count(context_window),
            pct,
            format_token_count(remaining),
        );
    }
    eprintln!(
        "  Input tokens:    {}  (cache hit: {}%)",
        format_token_count(total_in),
        snap.cache_hit_pct()
    );
    eprintln!(
        "  Output tokens:   {}",
        format_token_count(snap.output_tokens)
    );
    if snap.redactions > 0 {
        eprintln!(
            "  Redactions:      {} ({} image{}, ~{} MiB freed)",
            snap.redactions,
            snap.redacted_images,
            if snap.redacted_images == 1 { "" } else { "s" },
            snap.redacted_bytes / 1_048_576,
        );
    } else {
        eprintln!("  Redactions:      0");
    }
    eprintln!("  Messages:        {}", message_count);
}

/// Plain-text (no ANSI) rendering of account rate-limit usage, shared by the REPL/ACP `/usage`
/// command and the `meka account usage` CLI. Kept ANSI-free so the CLI can pipe it into scripts
/// unchanged; the trailing newline lets callers `print!`/`eprint!` it directly.
pub fn format_account_usage(usage: &crate::provider::AccountUsage) -> String {
    use std::fmt::Write as _;
    let mut out = String::from("Account usage\n");
    if usage.windows.is_empty() {
        out.push_str("  (no usage windows reported)\n");
    }
    for window in &usage.windows {
        let percent = window.used_percent.clamp(0.0, 100.0);
        let reset = window
            .resets_at
            .map(format_reset_time)
            .map(|when| format!("  (resets {when})"))
            .unwrap_or_default();
        let _ = writeln!(
            out,
            "  {:<18} {} {:>3}% used{}",
            window.label,
            usage_bar(percent),
            percent.round() as u64,
            reset,
        );
    }
    if let Some(extra) = &usage.extra_usage
        && let Some(line) = format_extra_usage(extra)
    {
        let _ = writeln!(out, "  Extra usage: {line}");
    }
    if let Some(note) = &usage.note {
        let _ = writeln!(out, "  {note}");
    }
    out
}

/// One-line summary of extra-usage / credits state, or `None` when there's nothing worth showing
/// (disabled with no balance and nothing spent).
fn format_extra_usage(extra: &crate::provider::ExtraUsage) -> Option<String> {
    let has_data =
        extra.enabled || extra.used.is_some_and(|used| used > 0.0) || extra.balance.is_some();
    if !has_data {
        return None;
    }
    let mut parts = vec![if extra.enabled { "enabled" } else { "disabled" }.to_string()];
    if let Some(utilization) = extra.utilization {
        parts.push(format!("{}% used", utilization.round() as i64));
    }
    if let Some(used) = extra.used {
        parts.push(format!(
            "{} spent",
            format_money(used, extra.currency.as_deref())
        ));
    }
    if let Some(balance) = extra.balance {
        parts.push(format!(
            "{} balance",
            format_money(balance, extra.currency.as_deref())
        ));
    }
    Some(parts.join(" · "))
}

/// Format a monetary amount: `$3.00` for USD/unknown, `3.00 EUR` otherwise.
fn format_money(amount: f64, currency: Option<&str>) -> String {
    match currency {
        Some("USD") | None => format!("${amount:.2}"),
        Some(other) => format!("{amount:.2} {other}"),
    }
}

/// REPL `/usage` rendering: the shared plain text to stderr (REPL UI feedback). The "not available"
/// case is handled by the caller via `render_hint`.
pub fn render_account_usage(usage: &crate::provider::AccountUsage) {
    eprint!("{}", format_account_usage(usage));
}

/// A fixed-width `[####------]` gauge for a 0-100 percentage.
fn usage_bar(percent: f64) -> String {
    const CELLS: usize = 10;
    let filled = ((percent / 100.0) * CELLS as f64).round() as usize;
    let filled = filled.min(CELLS);
    let mut bar = String::with_capacity(CELLS + 2);
    bar.push('[');
    for cell in 0..CELLS {
        bar.push(if cell < filled { '#' } else { '-' });
    }
    bar.push(']');
    bar
}

/// Format a non-negative duration in seconds compactly, e.g. `2d 3h`, `4h 12m`, `45m`, `30s`. Used
/// by `meka account whoami` for the token time-to-expiry.
pub(crate) fn format_duration_short(seconds: i64) -> String {
    let seconds = seconds.max(0);
    let minutes = seconds / 60;
    if minutes >= 24 * 60 {
        format!("{}d {}h", minutes / (24 * 60), (minutes % (24 * 60)) / 60)
    } else if minutes >= 60 {
        format!("{}h {}m", minutes / 60, minutes % 60)
    } else if minutes >= 1 {
        format!("{minutes}m")
    } else {
        format!("{seconds}s")
    }
}

/// Format a reset instant (Unix seconds) as "relative, local clock", e.g. "in 4h 12m, 2026-07-02
/// 02:10". Falls back to a plain clock when the timestamp is in the past or unparseable. Shared
/// with the ACP `/usage` text builder.
pub(crate) fn format_reset_time(epoch_seconds: i64) -> String {
    let Some(when) = chrono::DateTime::from_timestamp(epoch_seconds, 0) else {
        return "unknown".to_string();
    };
    let clock = when.with_timezone(&chrono::Local).format("%Y-%m-%d %H:%M");
    let minutes = when.signed_duration_since(chrono::Utc::now()).num_minutes();
    if minutes <= 0 {
        return format!("now, {clock}");
    }
    let relative = if minutes >= 24 * 60 {
        format!(
            "in {}d {}h",
            minutes / (24 * 60),
            (minutes % (24 * 60)) / 60
        )
    } else if minutes >= 60 {
        format!("in {}h {}m", minutes / 60, minutes % 60)
    } else {
        format!("in {minutes}m")
    };
    format!("{relative}, {clock}")
}

/// Print a single-line CLI error to stderr in the project's standard format.
pub fn render_error(error: &dyn std::fmt::Display) {
    eprintln!("{} {}", "Error:".with(Color::Red), error);
}

/// Print the "no provider configured" hint shown when the agent fails to initialize. Centralized so
/// the wording stays in sync everywhere.
pub fn render_provider_setup_hint() {
    eprintln!("Configure a provider to use meka.");
    eprintln!("Example: meka provider add work --type claude-oauth --model claude-opus-5");
    eprintln!("Run `meka provider list` to see configured profiles.");
}

/// Walk backwards through `messages` and return the suffix that starts at the `n`th most recent
/// user turn. A "turn" begins at a User-role message whose content is not purely `ToolResult`
/// blocks, i.e. an actual user prompt, not an agent-driven tool result echoed back as a User
/// message. `n == 0` or no qualifying turns returns an empty slice.
pub fn last_n_turns(
    messages: &[crate::provider::Message],
    n: usize,
) -> &[crate::provider::Message] {
    if n == 0 || messages.is_empty() {
        return &[];
    }
    // Walk backwards, tracking the earliest qualifying boundary seen so far. If we hit `n`
    // boundaries we stop there; if we exhaust the slice without reaching `n`, we return everything
    // from the earliest boundary we did find (so `N=999` on a 2-turn session still returns both
    // turns, not an empty slice).
    let mut seen = 0usize;
    let mut earliest_boundary: Option<usize> = None;
    for (index, message) in messages.iter().enumerate().rev() {
        if is_user_prompt_boundary(message) {
            seen += 1;
            earliest_boundary = Some(index);
            if seen == n {
                break;
            }
        }
    }
    match earliest_boundary {
        Some(start) => &messages[start..],
        None => &[],
    }
}

/// True when `message` is the start of a new turn from the user's perspective: Role::User with at
/// least one non-`ToolResult` block.
fn is_user_prompt_boundary(message: &crate::provider::Message) -> bool {
    use crate::provider::{ContentBlock, Role};
    if !matches!(message.role, Role::User) {
        return false;
    }
    message
        .content
        .iter()
        .any(|block| !matches!(block, ContentBlock::ToolResult { .. }))
}

/// Knobs for [`render_message_history`]. Mirrors the fields the live REPL reads off
/// `ResolvedConfig` so resumed/dumped history matches what the user sees during a live turn.
pub struct HistoryRenderOptions {
    pub render_mode: RenderMode,
    pub show_thinking: bool,
    pub input_style: nu_ansi_term::Style,
    /// Blank line before each user prompt (mirrors `[display].newline_before_prompt`).
    pub newline_before_prompt: bool,
    /// Blank line after each user prompt (mirrors `[display].newline_after_prompt`). Acts as the
    /// visual separator between the prompt and the agent's first response block.
    pub newline_after_prompt: bool,
}

/// Reprint a slice of historical messages styled to match the live REPL output. Inter-block spacing
/// flows through [`OutputSpacing`] (the same state machine the live loop uses) so transitions like
/// tool-indicator → text get a blank line; user-prompt spacing follows the `newline_before_prompt`
/// / `newline_after_prompt` config flags just like the live REPL.
pub fn render_message_history(messages: &[crate::provider::Message], opts: &HistoryRenderOptions) {
    use crate::provider::{ContentBlock, Role};
    if messages.is_empty() {
        return;
    }
    let mut spacing = OutputSpacing::new();
    // The caller (e.g. the `/history` dispatch) is expected to emit the leading blank, the
    // equivalent of the live REPL's `newline_after_prompt`, between its own command line and this
    // rendered history. So the very first user prompt we render must skip its own
    // `newline_before_prompt` to avoid stacking blanks. Once anything has been emitted, the inner
    // spacing rules take over and turn-to-turn transitions get their own blanks naturally.
    let mut emitted_any = false;
    for message in messages {
        for block in &message.content {
            match block {
                ContentBlock::Text { text } => match message.role {
                    Role::Assistant => {
                        if text.trim().is_empty() {
                            continue;
                        }
                        if spacing.before_text() {
                            eprintln!();
                        }
                        render_assistant_text(text, opts.render_mode);
                        emitted_any = true;
                    }
                    Role::User => {
                        let leading_blank = opts.newline_before_prompt && emitted_any;
                        if !render_user_prompt(text, opts.input_style, leading_blank) {
                            continue;
                        }
                        if opts.newline_after_prompt {
                            eprintln!();
                        }
                        spacing.after_prompt();
                        emitted_any = true;
                    }
                },
                // Input images (from an ACP client) have no terminal rendering; show a marker so a
                // replayed/exported transcript notes the attachment instead of dropping it
                // silently.
                ContentBlock::Image { .. } => {
                    if spacing.before_text() {
                        eprintln!();
                    }
                    eprintln!("[image]");
                    emitted_any = true;
                }
                ContentBlock::Thinking { thinking, .. } => {
                    if opts.show_thinking && !thinking.trim().is_empty() {
                        if spacing.before_thinking() {
                            eprintln!();
                        }
                        render_thinking_block(thinking, true);
                        emitted_any = true;
                    }
                }
                ContentBlock::RedactedThinking { .. } => {
                    if opts.show_thinking {
                        if spacing.before_thinking() {
                            eprintln!();
                        }
                        render_thinking_block("[redacted thinking]", true);
                        emitted_any = true;
                    }
                }
                ContentBlock::ToolUse { name, input, .. } => {
                    if spacing.before_tool_indicator() {
                        eprintln!();
                    }
                    render_tool_indicator(name, input, None);
                    emitted_any = true;
                }
                // Tool results are intentionally hidden; the live REPL doesn't echo them either,
                // so showing them in history would be a fidelity regression. The user sees the tool
                // indicator (above) and whatever the assistant's next text block says about the
                // result.
                ContentBlock::ToolResult { .. } => {}
            }
        }
    }
}

fn render_assistant_text(text: &str, render_mode: RenderMode) {
    // Caller has already emitted the leading blank line (via `OutputSpacing::before_text`) when
    // needed, and verified the text is non-empty. We just stream the markdown, no trailing blank,
    // because the next block's `before_*` will add one if appropriate.
    let mut renderer = StreamingRenderer::new(render_mode);
    if let Err(error) = renderer.push_delta(text) {
        tracing::debug!("history: failed to render assistant delta: {}", error);
    }
    if let Err(error) = renderer.finish() {
        tracing::debug!("history: failed to finish assistant render: {}", error);
    }
}

/// Render a user prompt with the cyan `>` gutter plus `input_style` applied to each line,
/// optionally preceded by a blank line. Returns `false` when the prompt was empty (after
/// `strip_context_tags`) and nothing was emitted, so the caller can skip the after-prompt
/// blank/state update.
fn render_user_prompt(text: &str, input_style: nu_ansi_term::Style, newline_before: bool) -> bool {
    let stripped = crate::session::strip_context_tags(text);
    let trimmed = stripped.trim();
    if trimmed.is_empty() {
        return false;
    }
    if newline_before {
        eprintln!();
    }
    for line in trimmed.lines() {
        eprintln!("{} {}", ">".with(Color::Cyan), input_style.paint(line));
    }
    true
}

pub fn render_thinking_block(thinking: &str, show_full: bool) {
    if show_full {
        eprintln!(
            "{}{}",
            "Thinking... ".with(Color::DarkGrey),
            thinking.with(Color::DarkGrey),
        );
    } else {
        let first_line = thinking.lines().next().unwrap_or("");
        let truncated = truncate_display(first_line, 80);
        eprintln!(
            "{}{}",
            "Thinking... ".with(Color::DarkGrey),
            truncated.with(Color::DarkGrey),
        );
    }
}

/// Render the todo list to stderr. Returns `true` if anything was printed, so the caller only
/// advances `OutputSpacing` when there was actually output; an empty list prints nothing and must
/// not claim a trailing blank line (otherwise the next text run loses its leading blank).
pub fn render_todo_list(title: Option<&str>, items: &[crate::tools::todo::TodoItem]) -> bool {
    use crate::tools::todo::TodoStatus;

    if items.is_empty() {
        return false;
    }
    eprintln!();

    // Heading is `TODO: <title>` (defensive fallback when absent), not indented, followed by a
    // blank line and the tasks as a markdown checklist.
    let heading = format!("TODO: {}", title.unwrap_or("Tasks"));
    eprintln!("{}", heading.with(Color::White).bold());
    eprintln!();

    for (index, item) in items.iter().enumerate() {
        let (marker, color) = match item.status {
            TodoStatus::Completed => ("[x]", Color::Green),
            TodoStatus::InProgress => ("[~]", Color::Yellow),
            TodoStatus::Pending => ("[ ]", Color::DarkGrey),
            TodoStatus::Cancelled => ("[-]", Color::DarkGrey),
        };
        let text = if item.status == TodoStatus::Cancelled {
            format!("(cancelled) {}", item.text)
        } else {
            item.text.clone()
        };
        eprintln!(
            "- {} {} {}",
            marker.with(color),
            (index + 1).to_string().with(Color::White),
            text
        );
    }

    eprintln!();
    true
}

pub fn tool_display_name_for_approval(name: &str) -> &str {
    tool_display_name(name)
}

/// Resolve the summary string shown next to a tool-call indicator and in the approval prompt. Tries
/// the hardcoded built-in map first; falls back to the tool's JSON schema `required[0]` when
/// provided (covers MCP tools, whose schemas are authored upstream and can't be enumerated here).
pub fn resolve_primary_param(
    name: &str,
    input: &serde_json::Value,
    schema: Option<&serde_json::Value>,
) -> Option<String> {
    if let Some(value) = builtin_primary_param(name, input) {
        return Some(value);
    }
    schema.and_then(|s| schema_primary_param(s, input))
}

fn tool_display_name(name: &str) -> &str {
    match name {
        "execute_command" => "Shell",
        "read_file" => "ReadFile",
        "write_file" => "WriteFile",
        "edit_file" => "EditFile",
        "find_files" => "FindFiles",
        "search_contents" => "SearchContents",
        "fetch_url" => "FetchUrl",
        "web_search" => "WebSearch",
        "todo" => "Todo",
        "spawn_agent" => "SpawnAgent",
        "scratchpad_write" => "ScratchpadWrite",
        "scratchpad_read" => "ScratchpadRead",
        "scratchpad_edit" => "ScratchpadEdit",
        "scratchpad_list" => "ScratchpadList",
        "scratchpad_delete" => "ScratchpadDelete",
        "skill" => "Skill",
        "render_image" => "RenderImage",
        other => other,
    }
}

fn builtin_primary_param(name: &str, input: &serde_json::Value) -> Option<String> {
    // `render_image` accepts either `from_scratchpad` or inline `base64`. Show the scratchpad name
    // when present; for inline base64 the payload is opaque so there's nothing useful to display.
    if name == "render_image" {
        if let Some(from) = input.get("from_scratchpad").and_then(|v| v.as_str()) {
            return Some(from.to_string());
        }
        if input.get("base64").is_some() {
            return Some("<inline base64>".to_string());
        }
        return None;
    }

    // `todo` has no single primary key. Surface what the agent is doing, preferring the status
    // transitions, then the `title` of a list it is building, then the list size, and finally
    // "read" for an argument-less read.
    if name == "todo" {
        if let Some(set) = input.get("set").and_then(|v| v.as_object()) {
            let parts: Vec<String> = set
                .iter()
                .filter_map(|(id, status)| {
                    status.as_str().map(|status| format!("#{} {}", id, status))
                })
                .collect();
            if !parts.is_empty() {
                return Some(parts.join(", "));
            }
        }
        if let Some(title) = input.get("title").and_then(|v| v.as_str()) {
            let title = title.trim();
            if !title.is_empty() {
                return Some(title.to_string());
            }
        }
        if let Some(items) = input.get("items").and_then(|v| v.as_array()) {
            let count = items.len();
            return Some(format!(
                "{} task{}",
                count,
                if count == 1 { "" } else { "s" }
            ));
        }
        return Some("read".to_string());
    }

    let key = match name {
        "execute_command" => "command",
        "read_file" | "write_file" | "edit_file" => "path",
        "find_files" | "search_contents" => "pattern",
        "fetch_url" => "url",
        "web_search" => "query",
        "spawn_agent" => "prompt",
        "scratchpad_write" | "scratchpad_read" | "scratchpad_edit" | "scratchpad_delete" => "name",
        "skill" => "name",
        _ => return None,
    };
    input.get(key).and_then(|v| v.as_str()).map(str::to_string)
}

/// Fallback for tools not covered by the built-in map (MCP tools, dynamically-registered tools,
/// etc.). Uses the first entry of `inputSchema.required` as the key into `input` and coerces the
/// value to a short display string. Returns `None` when the schema offers no `required` field, the
/// required key is missing from `input`, or the value type has no sensible string form (e.g. nested
/// objects / binary blobs).
fn schema_primary_param(schema: &serde_json::Value, input: &serde_json::Value) -> Option<String> {
    let required = schema.get("required")?.as_array()?;
    let key = required.iter().find_map(|v| v.as_str())?;
    let value = input.get(key)?;
    coerce_display_value(value)
}

fn coerce_display_value(value: &serde_json::Value) -> Option<String> {
    match value {
        serde_json::Value::String(s) => {
            if s.is_empty() {
                None
            } else {
                Some(s.clone())
            }
        }
        serde_json::Value::Number(n) => Some(n.to_string()),
        serde_json::Value::Bool(b) => Some(b.to_string()),
        serde_json::Value::Array(arr) => {
            let parts: Vec<String> = arr
                .iter()
                .filter_map(|v| match v {
                    serde_json::Value::String(s) if !s.is_empty() => Some(s.clone()),
                    serde_json::Value::Number(n) => Some(n.to_string()),
                    serde_json::Value::Bool(b) => Some(b.to_string()),
                    _ => None,
                })
                .collect();
            if parts.is_empty() {
                None
            } else {
                Some(parts.join(", "))
            }
        }
        _ => None,
    }
}

fn truncate_display(value: &str, max_chars: usize) -> String {
    let char_count = value.chars().count();
    if char_count <= max_chars {
        value.to_string()
    } else {
        let truncated: String = value.chars().take(max_chars).collect();
        format!("{}...", truncated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_duration_short() {
        assert_eq!(format_duration_short(30), "30s");
        assert_eq!(format_duration_short(90), "1m");
        assert_eq!(format_duration_short(3600 + 12 * 60), "1h 12m");
        assert_eq!(format_duration_short(2 * 86400 + 3 * 3600), "2d 3h");
        assert_eq!(format_duration_short(-5), "0s");
    }

    #[test]
    fn test_format_account_usage_is_ansi_free() {
        let usage = crate::provider::AccountUsage {
            windows: vec![crate::provider::UsageWindow {
                label: "5-hour (session)".into(),
                used_percent: 23.0,
                resets_at: None,
            }],
            extra_usage: None,
            note: None,
        };
        let out = format_account_usage(&usage);
        assert!(
            !out.contains('\u{1b}'),
            "must be ANSI-free for piping: {out:?}"
        );
        // Disabled/empty extra usage adds no line.
        assert!(!out.contains("Extra usage"), "got: {out:?}");
    }

    #[test]
    fn test_format_account_usage_shows_enabled_extra_usage() {
        let usage = crate::provider::AccountUsage {
            windows: vec![],
            extra_usage: Some(crate::provider::ExtraUsage {
                enabled: true,
                utilization: Some(70.0),
                used: Some(3.5),
                balance: Some(5.0),
                currency: None,
            }),
            note: None,
        };
        let out = format_account_usage(&usage);
        assert!(
            out.contains("Extra usage: enabled · 70% used · $3.50 spent · $5.00 balance"),
            "got: {out:?}"
        );
    }

    #[test]
    fn test_format_token_count_tiers() {
        assert_eq!(format_token_count(0), "0");
        assert_eq!(format_token_count(999), "999");
        assert_eq!(format_token_count(1_000), "1.0k");
        assert_eq!(format_token_count(234_000), "234.0k");
        assert_eq!(format_token_count(999_999), "1000.0k");
        assert_eq!(format_token_count(1_000_000), "1.0M");
        // A 1M-token model window renders compactly rather than as "1047.6k".
        assert_eq!(format_token_count(1_047_576), "1.0M");
        assert_eq!(format_token_count(2_300_000), "2.3M");
    }

    #[test]
    fn test_builtin_primary_param_todo() {
        // set transitions take priority.
        assert_eq!(
            builtin_primary_param(
                "todo",
                &serde_json::json!({ "title": "Build", "set": {"2": "in_progress"} })
            )
            .as_deref(),
            Some("#2 in_progress")
        );
        // title when building a list.
        assert_eq!(
            builtin_primary_param(
                "todo",
                &serde_json::json!({ "title": "Refactor auth", "items": ["a", "b", "c"] })
            )
            .as_deref(),
            Some("Refactor auth")
        );
        // items size as a fallback when there's no title.
        assert_eq!(
            builtin_primary_param("todo", &serde_json::json!({ "items": ["a", "b", "c"] }))
                .as_deref(),
            Some("3 tasks")
        );
        // empty argument-less call reads.
        assert_eq!(
            builtin_primary_param("todo", &serde_json::json!({})).as_deref(),
            Some("read")
        );
    }

    #[test]
    fn test_render_todo_list_reports_whether_it_rendered() {
        use crate::tools::todo::{TodoItem, TodoStatus};

        // Empty list prints nothing and must report `false` so the caller leaves spacing alone.
        assert!(!render_todo_list(None, &[]));

        let items = [TodoItem {
            text: "Do a thing".to_string(),
            status: TodoStatus::Pending,
        }];
        assert!(render_todo_list(Some("My list"), &items));
    }

    #[test]
    fn test_format_columns_aligns_and_leaves_last_unpadded() {
        let table = format_columns(&["Name", "Version", "Path"], &[
            vec!["a".to_string(), "1.0".to_string(), "/long/path".to_string()],
            vec![
                "longer-name".to_string(),
                "12".to_string(),
                "/p".to_string(),
            ],
        ]);
        let lines: Vec<&str> = table.lines().collect();
        assert_eq!(lines.len(), 3, "header + 2 rows");

        // The `Name` column widens to "longer-name" (11 chars); the header and the short row pad to
        // that width.
        assert!(lines[0].starts_with("Name         Version  Path"));
        assert!(lines[1].starts_with("a            1.0      /long/path"));
        assert!(lines[2].starts_with("longer-name  12       /p"));

        // The last column is never padded: no trailing whitespace.
        for line in &lines {
            assert_eq!(*line, line.trim_end(), "no trailing padding: {:?}", line);
        }
    }

    #[test]
    fn test_format_columns_empty_headers() {
        assert_eq!(format_columns(&[], &[]), "");
    }

    #[test]
    fn test_highlight_markdown_emits_ansi() {
        let out = highlight_markdown_to_string("# Hello\n");
        // ANSI escape prefix for any colored output.
        assert!(
            out.contains("\x1b["),
            "expected ANSI escape in highlighter output, got: {:?}",
            out
        );
        // Final reset so colors don't bleed into subsequent stdout writes.
        assert!(out.ends_with("\x1b[0m"));
    }

    #[test]
    fn test_highlight_markdown_preserves_content() {
        // Stripping ANSI escapes should give back the original text.
        let input = "Plain text with no markdown.\n";
        let out = highlight_markdown_to_string(input);
        let stripped = strip_ansi_escapes(&out);
        assert!(stripped.starts_with(input));
    }

    #[test]
    fn test_highlighter_uses_monokai_extended() {
        // Regression guard: the embedded theme file must parse and identify as Monokai Extended.
        // Catches accidental theme-file swaps or corrupted asset bytes at test time. Force OnceLock
        // init.
        let _ = highlight_markdown_to_string("");
        let theme = &highlighter().theme;
        assert_eq!(theme.name.as_deref(), Some("Monokai Extended"));
    }

    #[test]
    fn test_parse_fence_language() {
        assert_eq!(parse_fence_language("```rust"), Some("rust"));
        assert_eq!(parse_fence_language("```"), None);
        assert_eq!(parse_fence_language("```rust,ignore"), Some("rust"));
        assert_eq!(parse_fence_language("  ```python "), Some("python"));
        assert_eq!(parse_fence_language("```js title=x"), Some("js"));
    }

    #[test]
    fn test_syntax_for_language_resolves_and_falls_back() {
        assert_eq!(syntax_for_language(Some("rust")).name, "Rust");
        assert_eq!(syntax_for_language(Some("py")).name, "Python");
        // Absent / unknown tags fall back to the plain-text grammar rather than erroring.
        let plain = highlighter()
            .syntax_set
            .find_syntax_plain_text()
            .name
            .clone();
        assert_eq!(syntax_for_language(None).name, plain);
        assert_eq!(syntax_for_language(Some("nope-lang-xyz")).name, plain);
        // Alias: `shell`/`console` map to the shell grammar (not a syntect name/extension).
        assert!(
            syntax_for_language(Some("shell"))
                .name
                .to_ascii_lowercase()
                .contains("bash")
                || syntax_for_language(Some("shell"))
                    .name
                    .to_ascii_lowercase()
                    .contains("shell"),
            "shell alias resolved to {:?}",
            syntax_for_language(Some("shell")).name,
        );
    }

    #[test]
    fn test_code_block_body_is_language_highlighted() {
        // Regression guard for the whole feature: the Rust grammar tokenizes the body into several
        // colors, where the Markdown grammar (the old behavior) rendered it flat.
        let rust = "fn main() {\n    let x = 42;\n    println!(\"hi\");\n}\n";
        let rust_colors = distinct_fg_colors(&highlight_with_syntax(
            rust,
            syntax_for_language(Some("rust")),
        ));
        let markdown = highlighter()
            .syntax_set
            .find_syntax_by_name("Markdown")
            .expect("markdown syntax present");
        let flat_colors = distinct_fg_colors(&highlight_with_syntax(rust, markdown));
        assert!(
            rust_colors > 1,
            "rust body should be multi-colored, got {rust_colors}",
        );
        assert!(
            rust_colors > flat_colors,
            "language highlighting ({rust_colors}) should be richer than markdown ({flat_colors})",
        );
    }

    #[test]
    fn test_code_block_highlight_preserves_content() {
        // Stripping ANSI from a language-highlighted block returns the original code byte-for-byte.
        let code = "fn main() {\n    let x = 42;\n}\n";
        let out = highlight_with_syntax(code, syntax_for_language(Some("rust")));
        assert_eq!(strip_ansi_escapes(&out), code);
    }

    #[test]
    fn test_render_code_block_structure_and_body_highlight() {
        let lines = vec![
            "```rust".to_string(),
            "fn main() {".to_string(),
            "    let x = 42;".to_string(),
            "}".to_string(),
            "```".to_string(),
        ];
        let out = render_code_block_to_string(&lines);
        // Fences + body round-trip verbatim, each line ending in exactly one newline.
        assert_eq!(
            strip_ansi_escapes(&out),
            "```rust\nfn main() {\n    let x = 42;\n}\n```\n",
        );
        // The body is language-highlighted (multiple colors), not flat.
        assert!(distinct_fg_colors(&out) > 1);
    }

    #[test]
    fn test_render_code_block_handles_unterminated_and_empty() {
        // Unterminated block (no closing fence) still renders the opening fence + body.
        let unterminated = vec!["```rust".to_string(), "let x = 1;".to_string()];
        assert_eq!(
            strip_ansi_escapes(&render_code_block_to_string(&unterminated)),
            "```rust\nlet x = 1;\n",
        );
        // Empty block (fences only) renders both fences and no body.
        let empty = vec!["```rust".to_string(), "```".to_string()];
        assert_eq!(
            strip_ansi_escapes(&render_code_block_to_string(&empty)),
            "```rust\n```\n",
        );
        // No panic on the degenerate single-fence case.
        assert_eq!(
            strip_ansi_escapes(&render_code_block_to_string(&["```".to_string()])),
            "```\n",
        );
    }

    /// Count distinct 24-bit foreground colors (`ESC[38;2;R;G;B`) in ANSI-escaped output.
    fn distinct_fg_colors(ansi: &str) -> usize {
        let mut colors = std::collections::BTreeSet::new();
        let mut rest = ansi;
        while let Some(pos) = rest.find("\x1b[38;2;") {
            rest = &rest[pos + 7..];
            if let Some(end) = rest.find('m') {
                colors.insert(rest[..end].to_string());
            }
        }
        colors.len()
    }

    fn strip_ansi_escapes(input: &str) -> String {
        // Minimal CSI stripper for test assertions: drops `ESC [ ... letter`.
        let mut out = String::with_capacity(input.len());
        let mut chars = input.chars().peekable();
        while let Some(c) = chars.next() {
            if c == '\x1b' && chars.peek() == Some(&'[') {
                chars.next();
                for inner in chars.by_ref() {
                    if inner.is_ascii_alphabetic() {
                        break;
                    }
                }
            } else {
                out.push(c);
            }
        }
        out
    }

    #[test]
    fn test_truncate_display_short() {
        assert_eq!(truncate_display("hello", 10), "hello");
    }

    #[test]
    fn test_truncate_display_exact() {
        assert_eq!(truncate_display("hello", 5), "hello");
    }

    #[test]
    fn test_truncate_display_long() {
        assert_eq!(truncate_display("hello world", 5), "hello...");
    }

    #[test]
    fn test_truncate_display_empty() {
        assert_eq!(truncate_display("", 5), "");
    }

    #[test]
    fn test_tool_display_name_mappings() {
        assert_eq!(tool_display_name("execute_command"), "Shell");
        assert_eq!(tool_display_name("read_file"), "ReadFile");
        assert_eq!(tool_display_name("write_file"), "WriteFile");
        assert_eq!(tool_display_name("edit_file"), "EditFile");
        assert_eq!(tool_display_name("find_files"), "FindFiles");
        assert_eq!(tool_display_name("search_contents"), "SearchContents");
        assert_eq!(tool_display_name("fetch_url"), "FetchUrl");
        assert_eq!(tool_display_name("web_search"), "WebSearch");
        assert_eq!(tool_display_name("skill"), "Skill");
        assert_eq!(tool_display_name("render_image"), "RenderImage");
        assert_eq!(tool_display_name("custom_tool"), "custom_tool");
    }

    #[test]
    fn test_builtin_primary_param_skill() {
        let input = serde_json::json!({"name": "setup-postgres"});
        assert_eq!(
            builtin_primary_param("skill", &input).as_deref(),
            Some("setup-postgres")
        );
    }

    #[test]
    fn test_builtin_primary_param() {
        let input = serde_json::json!({"command": "ls", "path": "/tmp"});
        assert_eq!(
            builtin_primary_param("execute_command", &input).as_deref(),
            Some("ls")
        );
        assert_eq!(
            builtin_primary_param("read_file", &input).as_deref(),
            Some("/tmp")
        );
        assert_eq!(builtin_primary_param("unknown_tool", &input), None);
    }

    #[test]
    fn test_builtin_primary_param_missing() {
        let input = serde_json::json!({"other": "value"});
        assert_eq!(builtin_primary_param("execute_command", &input), None);
    }

    #[test]
    fn test_builtin_primary_param_render_image_from_scratchpad() {
        let input = serde_json::json!({"from_scratchpad": "frame4"});
        assert_eq!(
            builtin_primary_param("render_image", &input).as_deref(),
            Some("frame4")
        );
    }

    #[test]
    fn test_builtin_primary_param_render_image_inline_base64() {
        let input = serde_json::json!({"base64": "iVBOR..."});
        assert_eq!(
            builtin_primary_param("render_image", &input).as_deref(),
            Some("<inline base64>")
        );
    }

    #[test]
    fn test_builtin_primary_param_render_image_from_scratchpad_takes_precedence() {
        let input = serde_json::json!({"from_scratchpad": "frame4", "base64": "iVBOR..."});
        assert_eq!(
            builtin_primary_param("render_image", &input).as_deref(),
            Some("frame4")
        );
    }

    #[test]
    fn test_builtin_primary_param_render_image_empty() {
        let input = serde_json::json!({});
        assert_eq!(builtin_primary_param("render_image", &input), None);
    }

    #[test]
    fn test_schema_primary_param_string_value() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        });
        let input = serde_json::json!({"query": "best keyboards 2026"});
        assert_eq!(
            schema_primary_param(&schema, &input).as_deref(),
            Some("best keyboards 2026")
        );
    }

    #[test]
    fn test_schema_primary_param_array_of_strings() {
        let schema = serde_json::json!({
            "required": ["urls"],
        });
        let input = serde_json::json!({
            "urls": ["https://example.com", "https://other.example"],
        });
        assert_eq!(
            schema_primary_param(&schema, &input).as_deref(),
            Some("https://example.com, https://other.example")
        );
    }

    #[test]
    fn test_schema_primary_param_number_and_bool() {
        let schema = serde_json::json!({"required": ["count"]});
        let input = serde_json::json!({"count": 42});
        assert_eq!(schema_primary_param(&schema, &input).as_deref(), Some("42"));
        let schema = serde_json::json!({"required": ["enabled"]});
        let input = serde_json::json!({"enabled": true});
        assert_eq!(
            schema_primary_param(&schema, &input).as_deref(),
            Some("true")
        );
    }

    #[test]
    fn test_schema_primary_param_no_required_field() {
        let schema = serde_json::json!({
            "type": "object",
            "properties": {"query": {"type": "string"}},
        });
        let input = serde_json::json!({"query": "hello"});
        assert_eq!(schema_primary_param(&schema, &input), None);
    }

    #[test]
    fn test_schema_primary_param_required_key_absent_from_input() {
        let schema = serde_json::json!({"required": ["query"]});
        let input = serde_json::json!({"other_field": "value"});
        assert_eq!(schema_primary_param(&schema, &input), None);
    }

    #[test]
    fn test_schema_primary_param_empty_required_array() {
        let schema = serde_json::json!({"required": []});
        let input = serde_json::json!({"query": "hello"});
        assert_eq!(schema_primary_param(&schema, &input), None);
    }

    #[test]
    fn test_schema_primary_param_nested_object_skipped() {
        let schema = serde_json::json!({"required": ["config"]});
        let input = serde_json::json!({"config": {"nested": 1}});
        assert_eq!(schema_primary_param(&schema, &input), None);
    }

    #[test]
    fn test_resolve_primary_param_builtin_takes_precedence_over_schema() {
        // A tool that happens to share a built-in name: hardcoded map wins so the display stays
        // consistent with what users know.
        let schema = serde_json::json!({"required": ["path"]});
        let input = serde_json::json!({"command": "ls -la", "path": "/ignored"});
        assert_eq!(
            resolve_primary_param("execute_command", &input, Some(&schema)).as_deref(),
            Some("ls -la")
        );
    }

    #[test]
    fn test_resolve_primary_param_falls_back_to_schema_for_unknown_tool() {
        let schema = serde_json::json!({"required": ["query"]});
        let input = serde_json::json!({"query": "claude code"});
        assert_eq!(
            resolve_primary_param("exa__web_search_exa", &input, Some(&schema)).as_deref(),
            Some("claude code")
        );
    }

    #[test]
    fn test_resolve_primary_param_no_schema_no_builtin() {
        let input = serde_json::json!({"anything": "here"});
        assert_eq!(
            resolve_primary_param("unknown__mcp_tool", &input, None),
            None
        );
    }

    #[test]
    fn test_sanitize_strips_csi_and_c0() {
        // Clear-screen + home + bell, with ASCII text around.
        let input = "hello\x1b[2J\x1b[H\x07world\n";
        assert_eq!(sanitize_for_display(input), "helloworld\n");
    }

    #[test]
    fn test_sanitize_preserves_newline_tab_cr() {
        let input = "a\tb\nc\rd";
        assert_eq!(sanitize_for_display(input), "a\tb\nc\rd");
    }

    #[test]
    fn test_sanitize_strips_color_escape() {
        let input = "\x1b[31mred\x1b[0m";
        assert_eq!(sanitize_for_display(input), "red");
    }

    #[test]
    fn test_sanitize_strips_cursor_move() {
        let input = "\x1b[10;20H";
        assert_eq!(sanitize_for_display(input), "");
    }

    #[test]
    fn test_sanitize_preserves_unicode() {
        let input = "日本語 emoji \u{1F600}";
        assert_eq!(sanitize_for_display(input), "日本語 emoji \u{1F600}");
    }

    #[test]
    fn test_streaming_renderer_basic() {
        let mut renderer = StreamingRenderer::new(RenderMode::Termimad);
        renderer.push_delta("hello").unwrap();
        renderer.finish().unwrap();
    }

    #[test]
    fn test_streaming_renderer_strips_leading_newlines() {
        let mut renderer = StreamingRenderer::new(RenderMode::Termimad);
        renderer.push_delta("\n\nhello").unwrap();
        renderer.finish().unwrap();
    }

    #[test]
    fn test_render_mode_default() {
        assert_eq!(RenderMode::default(), RenderMode::Syntect);
    }

    #[test]
    fn test_render_mode_parses_syntect_only() {
        assert_eq!("syntect".parse(), Ok(RenderMode::Syntect));
        assert_eq!("rich".parse(), Ok(RenderMode::Termimad));
        assert_eq!(RenderMode::Syntect.to_string(), "syntect");
        // `bat` is no longer accepted (the alias was removed).
        assert!("bat".parse::<RenderMode>().is_err());
        assert!("nope".parse::<RenderMode>().is_err());
    }

    #[test]
    fn test_render_mode_config_rejects_bat() {
        assert_eq!(
            serde_json::from_str::<RenderMode>("\"syntect\"").unwrap(),
            RenderMode::Syntect,
        );
        // `render_mode = "bat"` no longer deserializes after the rename.
        assert!(serde_json::from_str::<RenderMode>("\"bat\"").is_err());
    }

    #[test]
    fn test_is_table_line() {
        assert!(is_table_line("| A | B |"));
        assert!(is_table_line("|---|---|"));
        assert!(is_table_line("| single |"));
        assert!(!is_table_line("|"));
        assert!(!is_table_line("not a table"));
        assert!(!is_table_line("| no trailing pipe"));
    }

    #[test]
    fn test_parse_table_row() {
        let cells = parse_table_row("| Alpha | Beta | Gamma |");
        assert_eq!(cells, vec!["Alpha", "Beta", "Gamma"]);
    }

    #[test]
    fn test_parse_table_row_no_spaces() {
        let cells = parse_table_row("|A|B|C|");
        assert_eq!(cells, vec!["A", "B", "C"]);
    }

    #[test]
    fn test_is_separator_row() {
        assert!(is_separator_row(&[
            "---".to_string(),
            "----".to_string(),
            "---".to_string()
        ]));
        assert!(is_separator_row(&[":--".to_string(), ":-:".to_string()]));
        assert!(!is_separator_row(&["Name".to_string(), "---".to_string()]));
    }

    #[test]
    fn test_format_table_alignment() {
        let lines = vec![
            "| Name | Value |".to_string(),
            "|------|-------|".to_string(),
            "| A | 100 |".to_string(),
            "| Beta | 2 |".to_string(),
        ];
        let result = format_table(&lines);
        assert_eq!(result.len(), 4);

        // All rows should have the same length
        let first_len = result[0].len();
        for (index, row) in result.iter().enumerate() {
            assert_eq!(
                row.len(),
                first_len,
                "row {} has length {} but expected {}",
                index,
                row.len(),
                first_len
            );
        }

        // Check content is padded
        assert_eq!(result[0], "| Name | Value |");
        assert_eq!(result[2], "| A    | 100   |");
        assert_eq!(result[3], "| Beta | 2     |");
    }

    #[test]
    fn test_format_table_wide_columns() {
        let lines = vec![
            "| # | Name | Type | Status | Score |".to_string(),
            "|---|------|------|--------|-------|".to_string(),
            "| 1 | Alpha | Primary | Pass | 98.5 |".to_string(),
            "| 2 | Beta | Secondary | Warn | 75.0 |".to_string(),
            "| 3 | Gamma | Primary | Pass | 91.2 |".to_string(),
        ];
        let result = format_table(&lines);
        let first_len = result[0].len();
        for (index, row) in result.iter().enumerate() {
            assert_eq!(
                row.len(),
                first_len,
                "row {} has length {} but expected {}",
                index,
                row.len(),
                first_len
            );
        }
    }

    #[test]
    fn test_format_table_empty() {
        let result = format_table(&[]);
        assert!(result.is_empty());
    }

    #[test]
    fn test_format_table_minimum_separator_width() {
        let lines = vec![
            "| A | B |".to_string(),
            "|---|---|".to_string(),
            "| C | D |".to_string(),
        ];
        let result = format_table(&lines);
        // Separator dashes should be at least 3 wide
        assert!(result[1].contains("---"));
    }

    #[test]
    fn test_format_table_emoji_single() {
        let lines = vec![
            "| Status | Name |".to_string(),
            "|---|---|".to_string(),
            "| 🟢 Pass | Alpha |".to_string(),
            "| 🔴 Fail | Beta |".to_string(),
        ];
        let result = format_table(&lines);
        assert_eq!(result.len(), 4);

        // All rows should have the same display width
        let first_width = display_width(&result[0]);
        for (index, row) in result.iter().enumerate() {
            assert_eq!(
                display_width(row),
                first_width,
                "row {} has display width {} but expected {}",
                index,
                display_width(row),
                first_width
            );
        }
    }

    #[test]
    fn test_format_table_emoji_multiple() {
        let lines = vec![
            "| Icon | Desc |".to_string(),
            "|---|---|".to_string(),
            "| 🟢🟢🟢 | Good |".to_string(),
            "| 🔴 | Bad |".to_string(),
        ];
        let result = format_table(&lines);
        let first_width = display_width(&result[0]);
        for (index, row) in result.iter().enumerate() {
            assert_eq!(
                display_width(row),
                first_width,
                "row {} has display width {} but expected {}",
                index,
                display_width(row),
                first_width
            );
        }
    }

    #[test]
    fn test_format_table_emoji_mixed_with_ascii() {
        let lines = vec![
            "| Segment | Change | Verdict |".to_string(),
            "|---|---|---|".to_string(),
            "| Canadian Banking | -9% | 🔴 Credit losses |".to_string(),
            "| Global Wealth | +17% | 🟢 AUM growth |".to_string(),
            "| Other | Flat | No emoji here |".to_string(),
        ];
        let result = format_table(&lines);
        let first_width = display_width(&result[0]);
        for (index, row) in result.iter().enumerate() {
            assert_eq!(
                display_width(row),
                first_width,
                "row {} has display width {} but expected {}",
                index,
                display_width(row),
                first_width
            );
        }
    }

    #[test]
    fn test_raw_mode_prints_text_verbatim() {
        let mut renderer = StreamingRenderer::new(RenderMode::Raw);
        renderer.push_delta("**bold** text\n").unwrap();
        renderer.finish().unwrap();
        // Raw mode just prints text as-is; if it didn't panic, it works
    }

    #[test]
    fn test_raw_mode_table_buffering() {
        let mut renderer = StreamingRenderer::new(RenderMode::Raw);
        renderer
            .push_delta("| A | B |\n|---|---|\n| C | D |\n\nafter table\n")
            .unwrap();
        renderer.finish().unwrap();
    }

    #[test]
    fn test_raw_mode_table_at_end() {
        let mut renderer = StreamingRenderer::new(RenderMode::Raw);
        renderer
            .push_delta("| A | B |\n|---|---|\n| C | D |")
            .unwrap();
        renderer.finish().unwrap();
    }

    #[test]
    fn test_finish_trims_trailing_newlines_raw() {
        let mut renderer = StreamingRenderer::new(RenderMode::Raw);
        renderer.push_delta("hello\n\n\n").unwrap();
        renderer.finish().unwrap();
    }

    #[test]
    fn test_finish_trims_trailing_newlines_rich() {
        let mut renderer = StreamingRenderer::new(RenderMode::Termimad);
        renderer.push_delta("hello\n\n\n").unwrap();
        renderer.finish().unwrap();
    }

    #[test]
    fn test_finish_only_newlines() {
        let mut renderer = StreamingRenderer::new(RenderMode::Raw);
        renderer.started = true;
        renderer.buffer = "\n\n\n".to_string();
        renderer.finish().unwrap();
    }

    #[test]
    fn test_normalize_spacing_adds_blank_line() {
        let input = "## Title\nBody text";
        let output = normalize_spacing(input);
        assert_eq!(output, "## Title\n\nBody text");
    }

    #[test]
    fn test_normalize_spacing_already_has_blank_line() {
        let input = "## Title\n\nBody text";
        let output = normalize_spacing(input);
        assert_eq!(output, "## Title\n\nBody text");
    }

    #[test]
    fn test_normalize_spacing_header_at_end() {
        let input = "## Title";
        let output = normalize_spacing(input);
        assert_eq!(output, "## Title");
    }

    #[test]
    fn test_normalize_spacing_inside_code_fence() {
        let input = "```\n## Not a header\ncode\n```";
        let output = normalize_spacing(input);
        assert_eq!(output, "```\n## Not a header\ncode\n```");
    }

    #[test]
    fn test_normalize_spacing_multiple_levels() {
        let input = "# H1\ntext\n### H3\nmore text";
        let output = normalize_spacing(input);
        assert_eq!(output, "# H1\n\ntext\n### H3\n\nmore text");
    }

    #[test]
    fn test_normalize_spacing_preserves_trailing_newline() {
        let input = "## Title\nBody\n";
        let output = normalize_spacing(input);
        assert_eq!(output, "## Title\n\nBody\n");
    }

    #[test]
    fn test_normalize_spacing_no_space_after_hash_is_not_header() {
        let input = "##not a header\ntext";
        let output = normalize_spacing(input);
        assert_eq!(output, "##not a header\ntext");
    }

    #[test]
    fn test_normalize_spacing_table_then_text() {
        let input = "| A | B |\n|---|---|\n| 1 | 2 |\n> blockquote";
        let output = normalize_spacing(input);
        assert_eq!(output, "| A | B |\n|---|---|\n| 1 | 2 |\n\n> blockquote");
    }

    #[test]
    fn test_normalize_spacing_table_already_has_blank_line() {
        let input = "| A | B |\n|---|---|\n| 1 | 2 |\n\n> blockquote";
        let output = normalize_spacing(input);
        assert_eq!(output, "| A | B |\n|---|---|\n| 1 | 2 |\n\n> blockquote");
    }

    #[test]
    fn test_normalize_spacing_table_inside_code_fence() {
        let input = "```\n| A | B |\n| 1 | 2 |\ncode\n```";
        let output = normalize_spacing(input);
        assert_eq!(output, "```\n| A | B |\n| 1 | 2 |\ncode\n```");
    }

    #[test]
    fn test_normalize_spacing_table_at_end() {
        let input = "| A | B |\n|---|---|\n| 1 | 2 |";
        let output = normalize_spacing(input);
        assert_eq!(output, "| A | B |\n|---|---|\n| 1 | 2 |");
    }

    use crate::provider::{ContentBlock, ImageSource, Message, Role, ToolResultContent};

    fn user_prompt(text: &str) -> Message {
        Message {
            role: Role::User,
            content: vec![ContentBlock::Text {
                text: text.to_string(),
            }],
        }
    }

    fn assistant_text(text: &str) -> Message {
        Message {
            role: Role::Assistant,
            content: vec![ContentBlock::Text {
                text: text.to_string(),
            }],
        }
    }

    fn tool_result_message(tool_use_id: &str, body: &str) -> Message {
        Message {
            role: Role::User,
            content: vec![ContentBlock::ToolResult {
                tool_use_id: tool_use_id.to_string(),
                content: vec![ToolResultContent::Text {
                    text: body.to_string(),
                }],
                is_error: false,
            }],
        }
    }

    #[test]
    fn test_last_n_turns_handles_empty() {
        assert!(last_n_turns(&[], 1).is_empty());
        assert!(last_n_turns(&[], 0).is_empty());
    }

    #[test]
    fn test_last_n_turns_zero_returns_empty() {
        let messages = vec![user_prompt("hi"), assistant_text("hello")];
        assert!(last_n_turns(&messages, 0).is_empty());
    }

    #[test]
    fn test_last_n_turns_one_counts_to_last_user_prompt() {
        let messages = vec![
            user_prompt("first"),
            assistant_text("ack one"),
            user_prompt("second"),
            assistant_text("ack two"),
        ];
        let slice = last_n_turns(&messages, 1);
        assert_eq!(slice.len(), 2);
        assert!(matches!(slice[0].role, Role::User));
        // The "second" prompt is the boundary; both messages after it belong to that turn.
        assert_eq!(
            slice[0].text_content(),
            "second",
            "boundary should be the most recent user prompt"
        );
    }

    #[test]
    fn test_last_n_turns_two_returns_from_earlier_boundary() {
        let messages = vec![
            user_prompt("first"),
            assistant_text("ack one"),
            user_prompt("second"),
            assistant_text("ack two"),
        ];
        let slice = last_n_turns(&messages, 2);
        assert_eq!(slice.len(), 4, "N=2 includes both turns end-to-end");
        assert_eq!(slice[0].text_content(), "first");
    }

    #[test]
    fn test_last_n_turns_n_exceeds_available_returns_all() {
        let messages = vec![user_prompt("only"), assistant_text("ack")];
        let slice = last_n_turns(&messages, 99);
        assert_eq!(slice.len(), 2);
        assert_eq!(slice[0].text_content(), "only");
    }

    #[test]
    fn test_last_n_turns_skips_tool_result_user_messages() {
        // A User message that's purely ToolResult blocks must not count as a turn boundary;
        // otherwise N=1 would land on the tool result echo instead of the user's actual prompt.
        let messages = vec![
            user_prompt("real prompt"),
            assistant_text("calling tool"),
            tool_result_message("toolu_1", "tool output"),
            assistant_text("answer"),
        ];
        let slice = last_n_turns(&messages, 1);
        assert_eq!(slice.len(), 4, "all messages belong to the one real turn");
        assert_eq!(slice[0].text_content(), "real prompt");
    }

    #[test]
    fn test_last_n_turns_no_user_prompt_returns_empty() {
        // Assistant-only history (rare; only happens if the materialised view starts
        // mid-conversation) has no turn boundaries; N doesn't find anything.
        let messages = vec![assistant_text("orphan reply")];
        assert!(last_n_turns(&messages, 1).is_empty());
    }

    #[test]
    fn test_is_user_prompt_boundary_classification() {
        assert!(is_user_prompt_boundary(&user_prompt("hi")));
        assert!(!is_user_prompt_boundary(&assistant_text("hi")));
        assert!(!is_user_prompt_boundary(&tool_result_message("u", "out")));

        // User message with mixed blocks (rare but possible) is still a boundary: at least one
        // block is not a ToolResult.
        let mixed = Message {
            role: Role::User,
            content: vec![
                ContentBlock::ToolResult {
                    tool_use_id: "u".to_string(),
                    content: vec![],
                    is_error: false,
                },
                ContentBlock::Text {
                    text: "follow-up".to_string(),
                },
            ],
        };
        assert!(is_user_prompt_boundary(&mixed));
    }

    #[test]
    fn test_render_message_history_does_not_panic_on_all_block_kinds() {
        // We can't capture stderr/stdout easily from a unit test, so we settle for "every variant
        // flows through without panicking".
        let messages = vec![
            user_prompt("can you read the file?"),
            Message {
                role: Role::Assistant,
                content: vec![
                    ContentBlock::Thinking {
                        thinking: "I should call read_file.".to_string(),
                        signature: None,
                    },
                    ContentBlock::RedactedThinking {
                        data: "opaque".to_string(),
                    },
                    ContentBlock::Text {
                        text: "Sure, reading now.".to_string(),
                    },
                    ContentBlock::ToolUse {
                        id: "u1".to_string(),
                        name: "read_file".to_string(),
                        input: serde_json::json!({"path": "a.txt"}),
                    },
                ],
            },
            Message {
                role: Role::User,
                content: vec![ContentBlock::ToolResult {
                    tool_use_id: "u1".to_string(),
                    content: vec![
                        ToolResultContent::Text {
                            text: "hello\n".to_string(),
                        },
                        ToolResultContent::Image {
                            source: ImageSource {
                                source_type: "base64".to_string(),
                                media_type: "image/png".to_string(),
                                data: "deadbeef".to_string(),
                            },
                        },
                    ],
                    is_error: false,
                }],
            },
            assistant_text("File starts with `hello`."),
        ];
        // Show-thinking on. If this panics, the test fails. We don't assert on captured output
        // (would need a TTY harness).
        let opts_with_thinking = HistoryRenderOptions {
            render_mode: RenderMode::Raw,
            show_thinking: true,
            input_style: nu_ansi_term::Style::default(),
            newline_before_prompt: true,
            newline_after_prompt: true,
        };
        render_message_history(&messages, &opts_with_thinking);
        // And off: the call must still complete cleanly.
        let opts_no_thinking = HistoryRenderOptions {
            show_thinking: false,
            ..opts_with_thinking
        };
        render_message_history(&messages, &opts_no_thinking);
        // Also: no-newline-prompt config must still produce non-panicking output.
        let opts_tight = HistoryRenderOptions {
            newline_before_prompt: false,
            newline_after_prompt: false,
            ..opts_with_thinking
        };
        render_message_history(&messages, &opts_tight);
    }

    #[test]
    fn test_render_message_history_empty_is_a_noop() {
        let opts = HistoryRenderOptions {
            render_mode: RenderMode::Raw,
            show_thinking: false,
            input_style: nu_ansi_term::Style::default(),
            newline_before_prompt: true,
            newline_after_prompt: true,
        };
        render_message_history(&[], &opts);
    }
}
